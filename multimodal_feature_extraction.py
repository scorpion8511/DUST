"""Pipeline for extracting multimodal image/text embeddings from a CSV manifest.

The script loads a paired image-text dataset described by a CSV file,
constructs crops based on optional patch-location columns, and feeds the
samples through a multimodal encoder to produce the feature tensors required by
``multimodal_scoring.py``.

The CSV file must provide, at minimum, the following columns:

* ``region_id`` – unique identifier for a spatial region (all magnifications of
  the same region share this value).
* ``magnification`` – numeric magnification level (e.g. ``5``/``10``/``20``).
* ``image_path`` – path to the source image (absolute or relative to
  ``--image-root``).
* ``text`` – paired text description for the region.

Optional columns describing patch geometry can also be supplied. By default the
script looks for ``patch_x``, ``patch_y``, and ``patch_size`` – interpreting the
value as a square crop. Width and height can be specified separately via
``patch_width`` and ``patch_height``. All column names are configurable via the
command-line flags documented in :func:`build_argparser`.

Model loading
-------------

The pipeline expects a TorchScript module with ``encode_image`` and
``encode_text`` methods. If the module exposes a ``tokenize`` helper it will be
used automatically; otherwise, pre-tokenised text can be supplied via an
additional CSV column (see ``--text-token-column``).

Outputs
-------

A ``.pth`` file containing three keys:

``image_embeddings``
    Mapping from magnification to a tensor of shape ``[N, D]``.

``text_embeddings``
    Tensor of shape ``[N, D]`` aligned with ``image_embeddings`` rows.

``metadata``
    Dictionary storing the region ids, raw text, and patch geometry for every
    entry. This metadata is meant to simplify downstream analysis and auditing
    of the feature extraction process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


DEFAULT_IMAGE_SIZE = 224
DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def _coerce_optional(value: Optional[str], cast) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return cast(value)


@dataclass
class PatchLocation:
    """Describes a rectangular crop within a source image."""

    x: float
    y: float
    width: Optional[float] = None
    height: Optional[float] = None

    def to_box(self, image_size: tuple[int, int], default_size: Optional[float]) -> tuple[int, int, int, int]:
        width = self.width if self.width is not None else default_size
        height = self.height if self.height is not None else default_size
        if width is None and height is None:
            raise ValueError("Patch dimensions are undefined; provide width/height or a default size.")
        if width is None:
            width = height
        if height is None:
            height = width
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))
        left = int(round(self.x))
        top = int(round(self.y))
        left = max(0, min(left, image_size[0] - 1))
        top = max(0, min(top, image_size[1] - 1))
        right = max(left + 1, min(left + width, image_size[0]))
        bottom = max(top + 1, min(top + height, image_size[1]))
        return left, top, right, bottom


@dataclass
class Sample:
    """Single image/text example drawn from the CSV manifest."""

    region_id: str
    csv_region_id: str
    magnification: int
    image_path: str
    text: Optional[str]
    text_tokens: Optional[List[int]]
    location: Optional[PatchLocation]
    raw_row: Mapping[str, str]


def _strip_magnification_suffix(identifier: str) -> str:
    """Collapse identifiers like ``patch_0_5x_10x`` to ``patch_0``."""

    parts = identifier.split("_")
    if len(parts) <= 1:
        return identifier

    base_parts = []
    for part in parts:
        token = part.strip().lower()
        if token.endswith("x"):
            magnitude = token[:-1]
            if magnitude.replace(".", "", 1).isdigit():
                continue
        base_parts.append(part)

    # Avoid stripping everything in pathological cases.
    if not base_parts:
        return identifier
    return "_".join(base_parts)


class ImageTextCSVDataset(Dataset):
    """Dataset that materialises samples from a CSV specification."""

    def __init__(
        self,
        csv_path: str,
        image_root: Optional[str],
        region_column: str,
        magnification_column: str,
        image_column: str,
        text_column: Optional[str],
        text_token_column: Optional[str],
        location_x_column: Optional[str],
        location_y_column: Optional[str],
        location_width_column: Optional[str],
        location_height_column: Optional[str],
        location_size_column: Optional[str],
        default_patch_size: Optional[float],
        strict_files: bool,
        strip_region_suffix: bool,
        filter_column: Optional[str],
        filter_values: Optional[Sequence[str]],
    ) -> None:
        self.samples: List[Sample] = []
        self._text_column = text_column
        self._text_token_column = text_token_column
        self._default_patch_size = default_patch_size
        image_root = os.path.expanduser(image_root) if image_root else None

        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = {region_column, magnification_column, image_column} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"CSV file is missing required columns: {', '.join(sorted(missing))}."
                )
            if filter_column and filter_column not in reader.fieldnames:
                raise ValueError(
                    f"Filter column '{filter_column}' is not present in the CSV header."
                )
            for row in reader:
                if filter_column:
                    value = row.get(filter_column)
                    if value is None:
                        continue
                    value = value.strip()
                    if filter_values is not None and value not in filter_values:
                        continue
                    if filter_values is None and value in {"", "0", "False", "false", "NONE", "None"}:
                        continue
                raw_region_id = str(row[region_column])
                region_id = (
                    _strip_magnification_suffix(raw_region_id)
                    if strip_region_suffix
                    else raw_region_id
                )
                try:
                    magnification = int(round(float(row[magnification_column])))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid magnification value {row[magnification_column]!r} for region {region_id}."
                    ) from exc

                image_path = row[image_column]
                if image_root and not os.path.isabs(image_path):
                    image_path = os.path.join(image_root, image_path)
                image_path = os.path.abspath(image_path)
                if strict_files and not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                text_value = row[text_column] if text_column else None
                text_value = text_value if text_value is None else text_value.strip()

                token_list: Optional[List[int]] = None
                if text_token_column:
                    raw_tokens = row.get(text_token_column, "")
                    raw_tokens = raw_tokens.strip()
                    if raw_tokens:
                        try:
                            if raw_tokens.startswith("["):
                                parsed = json.loads(raw_tokens)
                                if not isinstance(parsed, list):
                                    raise TypeError("Token column must encode a list of integers.")
                                token_list = [int(tok) for tok in parsed]
                            else:
                                token_list = [int(tok) for tok in raw_tokens.split()] if raw_tokens else []
                        except Exception as exc:  # pragma: no cover - defensive parsing
                            raise ValueError(
                                f"Failed to parse token sequence for region {region_id}: {raw_tokens!r}"
                            ) from exc

                loc_x = _coerce_optional(row.get(location_x_column) if location_x_column else None, float)
                loc_y = _coerce_optional(row.get(location_y_column) if location_y_column else None, float)
                loc_width = _coerce_optional(row.get(location_width_column) if location_width_column else None, float)
                loc_height = _coerce_optional(row.get(location_height_column) if location_height_column else None, float)
                loc_size = _coerce_optional(row.get(location_size_column) if location_size_column else None, float)
                location = None
                if loc_x is not None and loc_y is not None:
                    if loc_width is None and loc_height is None:
                        loc_width = loc_height = loc_size if loc_size is not None else default_patch_size
                    location = PatchLocation(x=loc_x, y=loc_y, width=loc_width, height=loc_height)

                sample = Sample(
                    region_id=region_id,
                    csv_region_id=raw_region_id,
                    magnification=magnification,
                    image_path=image_path,
                    text=text_value,
                    text_tokens=token_list,
                    location=location,
                    raw_row=row,
                )
                self.samples.append(sample)

        if not self.samples:
            raise ValueError("No samples were parsed from the CSV file.")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if sample.location is not None:
            box = sample.location.to_box(image.size, self._default_patch_size)
            image = image.crop(box)
            patch_metadata = {
                "x": sample.location.x,
                "y": sample.location.y,
                "width": sample.location.width,
                "height": sample.location.height,
                "box": box,
            }
        else:
            patch_metadata = None
        return {
            "region_id": sample.region_id,
            "magnification": sample.magnification,
            "image": image,
            "text": sample.text,
            "text_tokens": sample.text_tokens,
            "metadata": {
                "image_path": sample.image_path,
                "patch": patch_metadata,
                "csv_row": dict(sample.raw_row),
                "csv_region_id": sample.csv_region_id,
            },
        }


def _default_image_transform(size: int, mean: Sequence[float], std: Sequence[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _move_to_device(item, device: torch.device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, list):
        return [_move_to_device(elem, device) for elem in item]
    if isinstance(item, tuple):
        return tuple(_move_to_device(elem, device) for elem in item)
    if isinstance(item, Mapping):
        return {key: _move_to_device(val, device) for key, val in item.items()}
    return item


@dataclass
class TorchScriptEncoder:
    """Wrapper around a TorchScript module with CLIP-style encode methods."""

    path: str
    device: torch.device

    def __post_init__(self) -> None:
        module = torch.jit.load(self.path, map_location=self.device)
        module.eval()
        self.module = module

        if not hasattr(self.module, "encode_image") or not hasattr(self.module, "encode_text"):
            raise AttributeError(
                "TorchScript module must expose 'encode_image' and 'encode_text' methods."
            )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        return self.module.encode_image(images)

    def encode_text(
        self,
        texts: Optional[Sequence[Optional[str]]],
        tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if tokens is not None:
            tokens = tokens.to(self.device)
            return self.module.encode_text(tokens)

        if texts is None:
            raise ValueError("Text tokens must be supplied when the model lacks a tokenizer.")

        if hasattr(self.module, "tokenize"):
            processed = self.module.tokenize([text or "" for text in texts])
            processed = _move_to_device(processed, self.device)
            return self.module.encode_text(processed)

        return self.module.encode_text([text or "" for text in texts])


def build_collate_fn(image_transform: transforms.Compose):
    def collate(batch: Sequence[dict]) -> dict:
        images = [image_transform(item["image"]) for item in batch]
        image_tensor = torch.stack(images)

        texts = [item.get("text") for item in batch]
        if all(text is None for text in texts):
            text_list: Optional[List[Optional[str]]] = None
        else:
            text_list = texts

        token_seqs = [item.get("text_tokens") for item in batch]
        if any(seq is not None for seq in token_seqs):
            tensors = []
            for seq in token_seqs:
                if seq is None:
                    raise ValueError("Token column missing for a sample while others are provided.")
                tensors.append(torch.tensor(seq, dtype=torch.long))
            token_batch = pad_sequence(tensors, batch_first=True, padding_value=0)
        else:
            token_batch = None

        magnifications = torch.tensor([item["magnification"] for item in batch], dtype=torch.long)
        region_ids = [item["region_id"] for item in batch]
        metadata = [item["metadata"] for item in batch]

        return {
            "images": image_tensor,
            "texts": text_list,
            "tokens": token_batch,
            "magnifications": magnifications,
            "region_ids": region_ids,
            "metadata": metadata,
        }

    return collate


@dataclass
class RegionAccumulator:
    region_id: str
    text_embedding: Optional[torch.Tensor] = None
    text_value: Optional[str] = None
    images: Dict[int, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, object] = field(
        default_factory=lambda: {"patches": {}, "samples": [], "csv_region_ids": set()}
    )


def extract_features(
    dataloader: DataLoader,
    encoder: TorchScriptEncoder,
    magnifications: Optional[Sequence[int]],
    normalize: bool,
    drop_missing: bool,
) -> Dict[str, object]:
    regions: "OrderedDict[str, RegionAccumulator]" = OrderedDict()
    observed_magnifications: set[int] = set()

    device = encoder.device
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            tokens = batch["tokens"].to(device) if batch["tokens"] is not None else None
            texts = batch["texts"]
            image_embeddings = encoder.encode_image(images)
            text_embeddings = encoder.encode_text(texts, tokens)

            if normalize:
                image_embeddings = F.normalize(image_embeddings, dim=-1)
                text_embeddings = F.normalize(text_embeddings, dim=-1)

            for idx, region_id in enumerate(batch["region_ids"]):
                magnification = int(batch["magnifications"][idx].item())
                observed_magnifications.add(magnification)
                accumulator = regions.setdefault(region_id, RegionAccumulator(region_id=region_id))

                if accumulator.text_embedding is None:
                    accumulator.text_embedding = text_embeddings[idx].cpu()
                    accumulator.text_value = texts[idx] if texts is not None else None
                else:
                    existing = accumulator.text_embedding
                    new = text_embeddings[idx].cpu()
                    if torch.norm(existing - new).item() > 1e-4:
                        raise ValueError(
                            f"Multiple text embeddings detected for region {region_id} that do not match."
                        )

                accumulator.images[magnification] = image_embeddings[idx].cpu()
                accumulator.metadata.setdefault("samples", []).append(batch["metadata"][idx])
                patch = batch["metadata"][idx].get("patch")
                if patch is not None:
                    accumulator.metadata.setdefault("patches", {})[magnification] = patch
                original_id = batch["metadata"][idx].get("csv_region_id", region_id)
                accumulator.metadata.setdefault("csv_region_ids", set()).add(original_id)

    if magnifications is None:
        ordered_magnifications = sorted(observed_magnifications)
    else:
        ordered_magnifications = list(magnifications)
        missing = set(ordered_magnifications) - observed_magnifications
        if missing:
            raise ValueError(
                f"Requested magnifications {sorted(missing)} were not found in the dataset."
            )

    region_ids: List[str] = []
    csv_region_mapping: Dict[str, List[str]] = {}
    if not regions:
        raise ValueError("No regions were processed by the feature extractor.")

    text_embeddings = []
    raw_texts = []
    patches = {mag: [] for mag in ordered_magnifications}

    for region_id, accumulator in regions.items():
        if accumulator.text_embedding is None:
            if drop_missing:
                continue
            raise ValueError(f"Missing text embedding for region {region_id}.")
        text_embeddings.append(accumulator.text_embedding)
        raw_texts.append(accumulator.text_value)
        region_ids.append(region_id)
        csv_ids = accumulator.metadata.get("csv_region_ids", set())
        if not csv_ids:
            csv_ids = {region_id}
        csv_region_mapping[region_id] = sorted(csv_ids)
        for mag in ordered_magnifications:
            embedding = accumulator.images.get(mag)
            if embedding is None:
                if drop_missing:
                    break
                raise ValueError(
                    f"Region {region_id} does not contain an image for magnification {mag}."
                )
            patches[mag].append(accumulator.metadata.get("patches", {}).get(mag))
        else:
            continue
        # Only executed when break is triggered – remove partially appended data.
        text_embeddings.pop()
        raw_texts.pop()
        removed_id = region_ids.pop()
        csv_region_mapping.pop(removed_id, None)
        for mag in ordered_magnifications:
            if patches[mag]:
                patches[mag].pop()

    if not region_ids:
        raise ValueError(
            "All regions were filtered out – check the magnification coverage or disable --drop-missing."
        )

    text_tensor = torch.stack(text_embeddings, dim=0)

    image_tensors: Dict[int, torch.Tensor] = {}
    for mag in ordered_magnifications:
        embeddings = [regions[r].images.get(mag) for r in region_ids]
        if any(embed is None for embed in embeddings):
            missing_regions = [region_ids[idx] for idx, embed in enumerate(embeddings) if embed is None]
            raise ValueError(
                f"The following regions are missing magnification {mag}: {missing_regions}"
            )
        stacked = torch.stack(embeddings, dim=0)  # type: ignore[arg-type]
        image_tensors[mag] = stacked

    metadata = {
        "region_ids": region_ids,
        "texts": raw_texts,
        "patches": patches,
        "csv_region_mapping": csv_region_mapping,
    }

    return {
        "image_embeddings": image_tensors,
        "text_embeddings": text_tensor,
        "metadata": metadata,
        "magnifications": ordered_magnifications,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract multimodal features from a CSV manifest.")
    parser.add_argument("csv", type=str, help="Path to the CSV file with image/text pairs.")
    parser.add_argument("output", type=str, help="Destination .pth file for the extracted features.")

    parser.add_argument("--weights", type=str, required=True, help="Path to the TorchScript encoder weights.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (e.g. 'cpu' or 'cuda:0').")

    parser.add_argument("--image-root", type=str, default=None, help="Optional root to prepend to relative image paths.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Image resize square dimension.")
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=DEFAULT_MEAN,
        metavar=("R", "G", "B"),
        help="Normalization mean applied after resizing (default: CLIP mean).",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=DEFAULT_STD,
        metavar=("R", "G", "B"),
        help="Normalization std applied after resizing (default: CLIP std).",
    )

    parser.add_argument("--region-column", type=str, default="region_id")
    parser.add_argument("--magnification-column", type=str, default="magnification")
    parser.add_argument("--image-column", type=str, default="image_path")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument(
        "--text-token-column",
        type=str,
        default=None,
        help="Column containing pre-tokenised text (JSON list or space-delimited integers).",
    )
    parser.add_argument("--location-x", type=str, default="patch_x")
    parser.add_argument("--location-y", type=str, default="patch_y")
    parser.add_argument("--location-width", type=str, default="patch_width")
    parser.add_argument("--location-height", type=str, default="patch_height")
    parser.add_argument("--location-size", type=str, default="patch_size")
    parser.add_argument(
        "--default-patch-size",
        type=float,
        default=None,
        help="Fallback size (in pixels) when only X/Y are provided for a patch.",
    )
    parser.add_argument(
        "--filter-column",
        type=str,
        default=None,
        help="Optional column used to filter rows (e.g. 'is_test').",
    )
    parser.add_argument(
        "--filter-values",
        type=str,
        nargs="*",
        default=None,
        help="Values retained for --filter-column. Omit to drop falsy entries (''/0/False).",
    )
    parser.add_argument(
        "--magnifications",
        type=int,
        nargs="*",
        default=None,
        help="Expected magnifications (defaults to whatever is present).",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="Optional path to export metadata as JSON alongside the .pth file.",
    )
    parser.add_argument(
        "--strict-files",
        action="store_true",
        help="Fail if an image referenced in the CSV does not exist.",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Disable L2 normalisation of the extracted embeddings.",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop regions that are missing one or more requested magnifications instead of failing.",
    )
    parser.add_argument(
        "--strip-region-suffix",
        action="store_true",
        help="Normalize region identifiers by removing trailing magnification tokens like '_5x_10x'.",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    dataset = ImageTextCSVDataset(
        csv_path=args.csv,
        image_root=args.image_root,
        region_column=args.region_column,
        magnification_column=args.magnification_column,
        image_column=args.image_column,
        text_column=args.text_column,
        text_token_column=args.text_token_column,
        location_x_column=args.location_x,
        location_y_column=args.location_y,
        location_width_column=args.location_width,
        location_height_column=args.location_height,
        location_size_column=args.location_size,
        default_patch_size=args.default_patch_size,
        strict_files=args.strict_files,
        strip_region_suffix=args.strip_region_suffix,
        filter_column=args.filter_column,
        filter_values=args.filter_values,
    )

    transform = _default_image_transform(args.image_size, args.mean, args.std)
    collate_fn = build_collate_fn(transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device(args.device)
    encoder = TorchScriptEncoder(path=args.weights, device=device)

    features = extract_features(
        dataloader=dataloader,
        encoder=encoder,
        magnifications=args.magnifications,
        normalize=not args.skip_normalization,
        drop_missing=args.drop_missing,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "image_embeddings": features["image_embeddings"],
        "text_embeddings": features["text_embeddings"],
        "metadata": features["metadata"],
        "magnifications": features["magnifications"],
    }, args.output)

    if args.metadata_json:
        with open(args.metadata_json, "w", encoding="utf-8") as handle:
            json.dump(features["metadata"], handle, indent=2)

    print(f"Saved multimodal features to {args.output}")
    if args.metadata_json:
        print(f"Metadata exported to {args.metadata_json}")

    return features


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
