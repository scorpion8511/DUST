"""Feature extraction pipeline for PathGen CLIP-style encoders.

This script mirrors the CSV-driven workflow used by the PLIP, CONCH, and MUSK
helpers.  Given a manifest describing image/text pairs (optionally across
multiple magnifications per region), it loads a PathGen checkpoint using
``open_clip`` utilities, computes embeddings, and stores them in the
``.pth`` format consumed by ``multimodal_scoring.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multimodal_feature_extraction import ImageTextCSVDataset


def _maybe_login(token: Optional[str]) -> None:
    """Authenticate with Hugging Face when a token is supplied."""

    if not token:
        return
    try:  # pragma: no cover - runtime side effect
        from huggingface_hub import login

        login(token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover - provide useful diagnostics
        raise RuntimeError("Failed to authenticate with Hugging Face.") from exc


@dataclass
class RegionAccumulator:
    """Collect embeddings and metadata per region across magnifications."""

    region_id: str
    text_sum: Optional[torch.Tensor] = None
    text_count: int = 0
    text_values: List[str] = field(default_factory=list)
    images: Dict[int, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, object] = field(
        default_factory=lambda: {
            "patches": {},
            "samples": [],
            "csv_region_ids": set(),
            "base_region_ids": set(),
            "slide_ids": set(),
        }
    )
    labels: set[str] = field(default_factory=set)


def build_collate_fn(preprocess):
    """Create a collate function that applies the PathGen preprocess transform."""

    def collate(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        images = [preprocess(item["image"]) for item in batch]
        image_tensor = torch.stack(images, dim=0)

        texts = [item.get("text") for item in batch]
        if any(text is None for text in texts):
            raise ValueError(
                "Encountered rows without text descriptions; specify --text-column in the CSV."
            )

        magnifications = [int(item["magnification"]) for item in batch]
        region_ids = [str(item["region_id"]) for item in batch]
        metadata = [item["metadata"] for item in batch]

        return {
            "images": image_tensor,
            "texts": [text or "" for text in texts],
            "magnifications": magnifications,
            "region_ids": region_ids,
            "metadata": metadata,
        }

    return collate


def _ensure_iterable(value: Optional[Iterable[int]]) -> Optional[List[int]]:
    if value is None:
        return None
    return [int(v) for v in value]


def extract_pathgen_embeddings(
    dataloader: DataLoader,
    model,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool,
    magnifications: Optional[Sequence[int]],
    drop_missing: bool,
    label_column: Optional[str],
    slide_column: Optional[str],
) -> Dict[str, object]:
    regions: "OrderedDict[str, RegionAccumulator]" = OrderedDict()
    observed_magnifications: set[int] = set()

    model.eval()

    for batch in dataloader:
        images: torch.Tensor = batch["images"].to(device, dtype=dtype)
        texts: List[str] = batch["texts"]
        text_tokens = tokenizer(texts).to(device)

        with torch.inference_mode():
            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=dtype)
                if device.type == "cuda" and dtype == torch.float16
                else nullcontext()
            )
            with autocast_ctx:
                image_embeddings = model.encode_image(images)
                text_embeddings = model.encode_text(text_tokens)

        if normalize:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

        image_embeddings = image_embeddings.detach().cpu()
        text_embeddings = text_embeddings.detach().cpu()

        for idx, base_region_id in enumerate(batch["region_ids"]):
            magnification = int(batch["magnifications"][idx])
            observed_magnifications.add(magnification)

            csv_metadata = batch["metadata"][idx]
            csv_row = csv_metadata.get("csv_row")

            slide_value: Optional[str] = None
            if slide_column:
                if isinstance(csv_row, Mapping):
                    slide_value = csv_row.get(slide_column)  # type: ignore[index]
                if slide_value is None:
                    slide_value = csv_metadata.get(slide_column)
                if slide_value is not None:
                    slide_value = str(slide_value).strip()
                    if not slide_value:
                        slide_value = None

            if slide_value:
                region_key = f"{slide_value}::{base_region_id}"
            else:
                region_key = base_region_id

            accumulator = regions.setdefault(region_key, RegionAccumulator(region_id=region_key))

            embedding_image = image_embeddings[idx]
            embedding_text = text_embeddings[idx]

            if accumulator.text_sum is None:
                accumulator.text_sum = embedding_text.clone()
            else:
                accumulator.text_sum = accumulator.text_sum + embedding_text
            accumulator.text_count += 1
            accumulator.text_values.append(texts[idx])

            accumulator.images[magnification] = embedding_image
            accumulator.metadata.setdefault("samples", []).append(csv_metadata)
            accumulator.metadata.setdefault("base_region_ids", set()).add(base_region_id)
            if slide_value:
                accumulator.metadata.setdefault("slide_ids", set()).add(slide_value)

            patch = csv_metadata.get("patch")
            if patch is not None:
                accumulator.metadata.setdefault("patches", {})[magnification] = patch

            original_id = csv_metadata.get("csv_region_id", base_region_id)
            accumulator.metadata.setdefault("csv_region_ids", set()).add(original_id)

            if label_column:
                label_value = None
                if isinstance(csv_row, Mapping):
                    label_value = csv_row.get(label_column)
                if label_value is None:
                    label_value = csv_metadata.get(label_column)
                if label_value is not None:
                    if isinstance(label_value, str):
                        label_value = label_value.strip()
                    accumulator.labels.add(str(label_value))

    if magnifications is None:
        ordered_magnifications = sorted(observed_magnifications)
    else:
        ordered_magnifications = [int(m) for m in magnifications]
        missing = set(ordered_magnifications) - observed_magnifications
        if missing:
            raise ValueError(
                f"Requested magnifications {sorted(missing)} were not present in the dataset."
            )

    region_ids: List[str] = []
    text_embeddings: List[torch.Tensor] = []
    raw_texts: List[Optional[str]] = []
    csv_region_mapping: Dict[str, List[str]] = {}
    patches = {mag: [] for mag in ordered_magnifications}
    region_labels: List[Optional[str]] = []

    for region_id, accumulator in regions.items():
        if accumulator.text_sum is None or accumulator.text_count == 0:
            if drop_missing:
                continue
            raise ValueError(f"Missing text embedding for region {region_id}.")

        text_mean = accumulator.text_sum / float(accumulator.text_count)
        if normalize:
            text_mean = F.normalize(text_mean.unsqueeze(0), dim=-1).squeeze(0)

        region_ids.append(region_id)
        text_embeddings.append(text_mean)

        if accumulator.text_values:
            unique_texts = list(OrderedDict.fromkeys(accumulator.text_values))
            raw_texts.append(unique_texts[0] if len(unique_texts) == 1 else unique_texts)
        else:
            raw_texts.append(None)

        csv_ids = accumulator.metadata.get("csv_region_ids", set())
        if not csv_ids:
            csv_ids = {region_id}
        csv_region_mapping[region_id] = sorted(csv_ids)

        if label_column:
            label_values = sorted(accumulator.labels)
            if not label_values:
                region_labels.append(None)
            elif len(label_values) == 1:
                region_labels.append(label_values[0])
            else:
                raise ValueError(
                    f"Region {region_id} has conflicting labels {label_values} in column '{label_column}'."
                )

        for mag in ordered_magnifications:
            embedding = accumulator.images.get(mag)
            if embedding is None:
                if drop_missing:
                    break
                raise ValueError(
                    f"Region {region_id} is missing an image for magnification {mag}."
                )
            patches[mag].append(accumulator.metadata.get("patches", {}).get(mag))
        else:
            continue

        # Region skipped due to drop_missing -> undo partial appends.
        region_ids.pop()
        text_embeddings.pop()
        raw_texts.pop()
        csv_region_mapping.pop(region_id, None)
        for mag in ordered_magnifications:
            if patches[mag]:
                patches[mag].pop()
        if label_column and region_labels:
            region_labels.pop()

    if not region_ids:
        raise ValueError(
            "All regions were filtered out; disable --drop-missing or verify magnification coverage."
        )

    text_tensor = torch.stack(text_embeddings, dim=0)
    image_tensors: Dict[int, torch.Tensor] = {}
    for mag in ordered_magnifications:
        embeddings = [regions[r].images.get(mag) for r in region_ids]
        if any(embed is None for embed in embeddings):
            missing_regions = [region_ids[idx] for idx, embed in enumerate(embeddings) if embed is None]
            raise ValueError(
                f"The following regions lack magnification {mag}: {missing_regions}"
            )
        image_tensors[mag] = torch.stack(embeddings, dim=0)  # type: ignore[arg-type]

    base_region_aliases: Dict[str, List[str]] = {}
    slide_mapping: Dict[str, List[str]] = {}
    for region_id in region_ids:
        base_ids = regions[region_id].metadata.get("base_region_ids", set())
        if base_ids:
            base_region_aliases[region_id] = sorted(
                str(alias) for alias in base_ids if alias is not None
            )
        slide_ids = regions[region_id].metadata.get("slide_ids", set())
        if slide_ids:
            slide_mapping[region_id] = sorted(str(slide) for slide in slide_ids if slide is not None)

    metadata = {
        "region_ids": region_ids,
        "texts": raw_texts,
        "patches": patches,
        "csv_region_mapping": csv_region_mapping,
    }
    if base_region_aliases:
        metadata["base_region_ids"] = base_region_aliases
    if slide_mapping:
        metadata["slide_ids"] = slide_mapping
    if label_column:
        metadata["labels"] = region_labels
        metadata["label_column"] = label_column

    return {
        "image_embeddings": image_tensors,
        "text_embeddings": text_tensor,
        "metadata": metadata,
        "magnifications": ordered_magnifications,
    }


def save_outputs(features: Mapping[str, object], output_path: str, metadata_json: Optional[str]) -> None:
    torch.save(features, output_path)
    if metadata_json:
        serialisable = {
            "region_ids": features["metadata"]["region_ids"],
            "texts": features["metadata"]["texts"],
            "patches": features["metadata"]["patches"],
            "csv_region_mapping": features["metadata"]["csv_region_mapping"],
            "magnifications": features["magnifications"],
        }
        if "base_region_ids" in features["metadata"]:
            serialisable["base_region_ids"] = features["metadata"]["base_region_ids"]
        if "slide_ids" in features["metadata"]:
            serialisable["slide_ids"] = features["metadata"]["slide_ids"]
        if "labels" in features["metadata"]:
            serialisable["labels"] = features["metadata"]["labels"]
            serialisable["label_column"] = features["metadata"].get("label_column")
        with open(metadata_json, "w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract PathGen features from a CSV manifest.")
    parser.add_argument("csv", type=str, help="Path to the CSV file with image/text pairs.")
    parser.add_argument("output", type=str, help="Destination .pth file for the extracted features.")

    parser.add_argument(
        "--model", type=str, default="ViT-B-16", help="Model backbone passed to open_clip.create_model_and_transforms."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Checkpoint identifier or path passed to open_clip.create_model_and_transforms.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference (default: cuda if available else cpu).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Number of samples per DataLoader batch.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Floating point precision for image features (fp16 enables autocast on CUDA).",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable L2 normalisation of the embeddings.",
    )
    parser.set_defaults(normalize=True)

    parser.add_argument("--image-root", type=str, default=None, help="Optional root to prepend to relative image paths.")
    parser.add_argument("--region-column", type=str, default="region_id")
    parser.add_argument("--magnification-column", type=str, default="magnification")
    parser.add_argument("--image-column", type=str, default="image_path")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--text-token-column", type=str, default=None)
    parser.add_argument("--patch-x-column", type=str, default=None)
    parser.add_argument("--patch-y-column", type=str, default=None)
    parser.add_argument("--patch-width-column", type=str, default=None)
    parser.add_argument("--patch-height-column", type=str, default=None)
    parser.add_argument("--patch-size-column", type=str, default=None)
    parser.add_argument(
        "--default-patch-size",
        type=float,
        default=None,
        help="Fallback square size when only coordinates are supplied.",
    )
    parser.add_argument("--strict-files", action="store_true", help="Raise if image files referenced in the CSV are missing.")
    parser.add_argument(
        "--strip-region-suffix",
        action="store_true",
        help="Remove trailing magnification suffixes when grouping regions (e.g. patch_0_5x → patch_0).",
    )
    parser.add_argument("--filter-column", type=str, default=None)
    parser.add_argument(
        "--filter-values",
        type=str,
        nargs="*",
        default=None,
        help="Retain rows where the filter column matches one of the specified values.",
    )
    parser.add_argument(
        "--magnifications",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of magnifications to retain; defaults to those discovered in the CSV.",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Discard regions missing any requested magnification instead of raising an error.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="CSV column containing region-level labels to store in the output metadata.",
    )
    parser.add_argument(
        "--slide-column",
        type=str,
        default=None,
        help="CSV column identifying the slide for each region (used to disambiguate region IDs).",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="Optional path to export the metadata dictionary as JSON alongside the .pth archive.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for gated checkpoints (login is skipped when unset).",
    )

    return parser


def load_pathgen_model(args: argparse.Namespace):
    _maybe_login(args.hf_token)

    import open_clip

    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
    )

    tokenizer = open_clip.get_tokenizer(args.model)

    model = model.to(device, dtype=dtype).eval()

    return model, preprocess, tokenizer, device, dtype


def run(args: argparse.Namespace) -> None:
    dataset = ImageTextCSVDataset(
        csv_path=args.csv,
        image_root=args.image_root,
        region_column=args.region_column,
        magnification_column=args.magnification_column,
        image_column=args.image_column,
        text_column=args.text_column,
        text_token_column=args.text_token_column,
        location_x_column=args.patch_x_column,
        location_y_column=args.patch_y_column,
        location_width_column=args.patch_width_column,
        location_height_column=args.patch_height_column,
        location_size_column=args.patch_size_column,
        default_patch_size=args.default_patch_size,
        strict_files=args.strict_files,
        strip_region_suffix=args.strip_region_suffix,
        filter_column=args.filter_column,
        filter_values=args.filter_values,
    )

    model, preprocess, tokenizer, device, dtype = load_pathgen_model(args)

    collate_fn = build_collate_fn(preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    features = extract_pathgen_embeddings(
        dataloader=dataloader,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        normalize=args.normalize,
        magnifications=_ensure_iterable(args.magnifications),
        drop_missing=args.drop_missing,
        label_column=args.label_column,
        slide_column=args.slide_column,
    )

    save_outputs(features, args.output, args.metadata_json)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
