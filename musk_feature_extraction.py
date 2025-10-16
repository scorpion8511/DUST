"""Extract MUSK multimodal features from a CSV manifest.

This script mirrors the official MUSK demo pipeline while scaling it to an
entire dataset of paired image/text patches. Given a CSV file describing the
patches, it

1. loads the MUSK vision-language encoder via ``timm.create_model``;
2. crops and normalises every image region using the MUSK preprocessing stack;
3. tokenises each region description with the repository-provided
   ``xlm_tokenizer`` helper; and
4. stores per-magnification image embeddings together with the aligned text
   embeddings and metadata so that ``multimodal_scoring.py`` can evaluate MSCI
   and CMI-LB.

The resulting ``.pth`` file contains the keys required by the scoring
utilities::

    {
        "image_embeddings": {5: tensor[N, D], 10: ..., 20: ...},
        "text_embeddings": tensor[N, D],
        "metadata": {...}
    }

Optional JSON metadata can be exported with ``--metadata-json`` to aid manual
inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from multimodal_feature_extraction import ImageTextCSVDataset


# ---------------------------------------------------------------------------
# Optional progress bar (falls back to a simple iterator if tqdm is missing).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - best-effort import
    tqdm = None  # type: ignore


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ensure_fairscale_stub() -> None:
    """Provide minimal ``fairscale.nn`` symbols if the package is absent."""

    if "fairscale" in sys.modules:
        return

    module = type(sys)("fairscale")
    nn_module = type(sys)("fairscale.nn")

    def _identity(x, *args, **kwargs):  # pragma: no cover - trivial fallback
        return x

    nn_module.checkpoint_wrapper = _identity
    nn_module.wrap = _identity

    module.nn = nn_module  # type: ignore[attr-defined]
    sys.modules["fairscale"] = module
    sys.modules["fairscale.nn"] = nn_module


def _resolve_repo_root(path: Optional[str]) -> Optional[str]:
    if not path:
        return None

    current = os.path.abspath(path)
    for _ in range(4):  # search upwards a few levels for the package root
        candidate = os.path.join(current, "musk", "__init__.py")
        if os.path.isfile(candidate):
            return current
        current = os.path.dirname(current)
    return os.path.abspath(path)


def _default_tokenizer_path(repo_root: Optional[str]) -> Optional[str]:
    if not repo_root:
        return None
    candidate = os.path.join(repo_root, "musk", "models", "tokenizer.spm")
    return candidate if os.path.isfile(candidate) else None


def _dtype_from_precision(precision: str) -> torch.dtype:
    precision = precision.lower()
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision in {"fp32", "float", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported precision '{precision}'.")


def _maybe_login(token: Optional[str]) -> None:
    if not token:
        return
    try:  # pragma: no cover - runtime side effect
        from huggingface_hub import login

        login(token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover - informative failure
        raise RuntimeError("Failed to authenticate with Hugging Face.") from exc


def _build_transform(image_size: int, antialias: bool) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=antialias),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    )


@dataclass
class RegionState:
    """Book-keeping for a single region aggregated across magnifications."""

    csv_region_ids: set = field(default_factory=set)
    texts: List[str] = field(default_factory=list)
    text_embedding: Optional[Tensor] = None
    magnification_embeddings: Dict[int, Tensor] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=lambda: {"patches": {}, "rows": []})
    base_region_ids: set = field(default_factory=set)
    slide_ids: set = field(default_factory=set)
    labels: set[str] = field(default_factory=set)


class MUSKEncoder:
    """Thin wrapper around the official MUSK timm model."""

    def __init__(
        self,
        model_name: str,
        checkpoint: str,
        device: str,
        precision: str,
        ms_augment: bool,
        tokenizer_path: str,
        text_max_length: int,
        repo_root: Optional[str] = None,
        strict_checkpoint_key: str = "model|module",
    ) -> None:
        repo_root = _resolve_repo_root(repo_root)
        if repo_root:
            sys.path.insert(0, repo_root)

        _ensure_fairscale_stub()

        try:
            from musk import modeling as _musk_modeling  # type: ignore  # noqa: F401
            from musk import utils  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime feedback
            raise ImportError(
                "Could not import the 'musk' package. Provide --musk-repo pointing to a clone of"
                " https://github.com/lilab-stanford/MUSK or install it via 'pip install -e .'"
            ) from exc

        from timm.models import create_model

        self.device = torch.device(device)
        self.dtype = _dtype_from_precision(precision)
        self.ms_augment = ms_augment
        self.text_max_length = int(text_max_length)
        self.utils = utils

        self.model = create_model(model_name).eval()
        utils.load_model_and_may_interpolate(checkpoint, self.model, strict_checkpoint_key, "")
        self.model.to(self.device, dtype=self.dtype).eval()

        from transformers import XLMRobertaTokenizer

        self.tokenizer = XLMRobertaTokenizer(tokenizer_path, use_fast=False)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = "<pad>"

    def encode_images(self, images: Tensor) -> Tensor:
        images = images.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            outputs = self.model(
                image=images,
                with_head=True,
                out_norm=True,
                ms_aug=self.ms_augment,
                return_global=True,
            )
        image_embeddings = outputs[0]
        return image_embeddings.float().cpu()

    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        ids_list: List[Tensor] = []
        pad_list: List[Tensor] = []
        for text in texts:
            if text is None:
                raise ValueError("Encountered missing caption for a region; supply --text-column with text data.")
            token_ids, padding = self.utils.xlm_tokenizer(text, self.tokenizer, max_len=self.text_max_length)
            token_tensor = torch.tensor(token_ids, dtype=torch.long)
            pad_tensor = torch.tensor(padding, dtype=torch.bool)
            if token_tensor.ndim == 1:
                token_tensor = token_tensor.unsqueeze(0)
            if pad_tensor.ndim == 1:
                pad_tensor = pad_tensor.unsqueeze(0)
            ids_list.append(token_tensor)
            pad_list.append(pad_tensor)

        text_ids = torch.cat(ids_list, dim=0).to(self.device)
        padding_mask = torch.cat(pad_list, dim=0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                text_description=text_ids,
                padding_mask=padding_mask,
                with_head=True,
                out_norm=True,
                ms_aug=False,
                return_global=True,
            )
        text_embeddings = outputs[1]
        return text_embeddings.float().cpu()


def _collate_samples(batch: Sequence[Mapping[str, object]]) -> Dict[str, List[object]]:
    return {
        "images": [item["image"] for item in batch],
        "texts": [item.get("text") for item in batch],
        "magnifications": [int(item["magnification"]) for item in batch],
        "region_ids": [str(item["region_id"]) for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def _accumulate_metadata(
    entry: RegionState,
    metadata: Mapping[str, object],
    magnification: int,
    base_region_id: str,
    slide_value: Optional[str],
) -> None:
    csv_region_id = metadata.get("csv_region_id", base_region_id)
    entry.csv_region_ids.add(str(csv_region_id))
    entry.metadata.setdefault("rows", []).append(metadata.get("csv_row"))
    entry.metadata.setdefault("images", {})[magnification] = metadata.get("image_path")
    if metadata.get("patch") is not None:
        entry.metadata.setdefault("patches", {})[magnification] = metadata.get("patch")
    entry.base_region_ids.add(str(base_region_id))
    entry.metadata.setdefault("base_region_ids", set()).add(str(base_region_id))
    if slide_value:
        entry.slide_ids.add(slide_value)
        entry.metadata.setdefault("slide_ids", set()).add(slide_value)


def extract_musk_embeddings(
    csv_path: str,
    output_path: str,
    encoder: MUSKEncoder,
    image_transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    dataset_kwargs: Mapping[str, object],
    metadata_json: Optional[str] = None,
    label_column: Optional[str] = None,
    slide_column: Optional[str] = None,
) -> None:
    dataset = ImageTextCSVDataset(csv_path=csv_path, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_samples,
    )

    regions: "OrderedDict[str, RegionState]" = OrderedDict()

    iterator: Iterable[Mapping[str, object]]
    if tqdm is not None:
        iterator = tqdm(loader, desc="Extracting", unit="batch")
    else:
        iterator = loader

    for batch in iterator:
        pil_images = batch["images"]
        image_tensors = torch.stack([image_transform(img) for img in pil_images])
        image_embeddings = encoder.encode_images(image_tensors)

        for idx, base_region_id in enumerate(batch["region_ids"]):
            magnification = batch["magnifications"][idx]
            text_value = batch["texts"][idx]
            metadata = batch["metadata"][idx]
            csv_row = metadata.get("csv_row")

            slide_value: Optional[str] = None
            if slide_column:
                if isinstance(csv_row, Mapping):
                    slide_value = csv_row.get(slide_column)  # type: ignore[index]
                if slide_value is None:
                    slide_value = metadata.get(slide_column)
                if slide_value is not None:
                    slide_value = str(slide_value).strip()
                    if not slide_value:
                        slide_value = None

            if slide_value:
                region_key = f"{slide_value}::{base_region_id}"
            else:
                region_key = base_region_id

            entry = regions.setdefault(region_key, RegionState())
            _accumulate_metadata(entry, metadata, magnification, str(base_region_id), slide_value)
            if text_value and not entry.texts:
                entry.texts.append(text_value)

            if label_column:
                label_value = None
                if isinstance(csv_row, Mapping):
                    label_value = csv_row.get(label_column)
                if label_value is None:
                    label_value = metadata.get(label_column)
                if label_value is not None:
                    if isinstance(label_value, str):
                        label_value = label_value.strip()
                    entry.labels.add(str(label_value))

            entry.magnification_embeddings[magnification] = image_embeddings[idx]

    # Encode unique texts once all regions have been seen.
    pending_regions = [rid for rid, state in regions.items() if state.text_embedding is None and state.texts]
    if pending_regions:
        texts = [regions[rid].texts[0] for rid in pending_regions]
        text_embeddings = encoder.encode_texts(texts)
        for rid, embedding in zip(pending_regions, text_embeddings):
            regions[rid].text_embedding = embedding

    magnifications = sorted({mag for state in regions.values() for mag in state.magnification_embeddings})
    if not magnifications:
        raise RuntimeError("No magnifications were processed; ensure the CSV contains valid data.")

    image_outputs: Dict[int, List[Tensor]] = {mag: [] for mag in magnifications}
    text_outputs: List[Tensor] = []
    kept_regions: List[str] = []
    per_region_metadata: List[MutableMapping[str, object]] = []
    region_labels: List[Optional[str]] = []
    base_region_aliases: Dict[str, List[str]] = {}
    slide_mapping: Dict[str, List[str]] = {}

    for region_id, state in regions.items():
        if state.text_embedding is None:
            continue
        if any(mag not in state.magnification_embeddings for mag in magnifications):
            continue

        kept_regions.append(region_id)
        text_outputs.append(state.text_embedding)

        label_value: Optional[str] = None
        if label_column:
            label_values = sorted(state.labels)
            if not label_values:
                label_value = None
            elif len(label_values) == 1:
                label_value = label_values[0]
            else:
                raise ValueError(
                    f"Region {region_id} has conflicting labels {label_values} in column '{label_column}'."
                )
            region_labels.append(label_value)

        for mag in magnifications:
            image_outputs[mag].append(state.magnification_embeddings[mag])

        metadata_entry: MutableMapping[str, object] = {
            "region_id": region_id,
            "csv_region_ids": sorted(state.csv_region_ids),
            "text": state.texts[0] if state.texts else None,
            "images": state.metadata.get("images", {}),
            "patches": state.metadata.get("patches", {}),
            "rows": state.metadata.get("rows", []),
        }
        if state.base_region_ids:
            aliases = sorted(str(alias) for alias in state.base_region_ids if alias is not None)
            metadata_entry["base_region_ids"] = aliases
            base_region_aliases[region_id] = aliases
        if state.slide_ids:
            slides = sorted(str(slide) for slide in state.slide_ids if slide is not None)
            metadata_entry["slide_ids"] = slides
            slide_mapping[region_id] = slides
        if label_column:
            metadata_entry["label"] = label_value
        per_region_metadata.append(metadata_entry)

    if not kept_regions:
        raise RuntimeError("No regions contained both text and all requested magnifications.")

    image_embeddings_stacked = {mag: torch.stack(v, dim=0) for mag, v in image_outputs.items()}
    text_embeddings_stacked = torch.stack(text_outputs, dim=0)

    payload = {
        "image_embeddings": {mag: tensor for mag, tensor in image_embeddings_stacked.items()},
        "text_embeddings": text_embeddings_stacked,
        "metadata": {
            "regions": kept_regions,
            "magnifications": magnifications,
            "per_region": per_region_metadata,
        },
    }

    if base_region_aliases:
        payload["metadata"]["base_region_ids"] = base_region_aliases
    if slide_mapping:
        payload["metadata"]["slide_ids"] = slide_mapping
    if label_column:
        payload["metadata"]["labels"] = region_labels
        payload["metadata"]["label_column"] = label_column

    torch.save(payload, output_path)

    if metadata_json:
        with open(metadata_json, "w", encoding="utf-8") as handle:
            json.dump(payload["metadata"], handle, indent=2)


# ---------------------------------------------------------------------------
# Argument parsing and CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MUSK embeddings for MSCI/CMI-LB scoring.")
    parser.add_argument("csv", help="CSV manifest describing the image/text pairs.")
    parser.add_argument("output", help="Destination .pth file for the extracted features.")

    parser.add_argument("--image-root", default=None, help="Root directory to prepend to relative image paths.")
    parser.add_argument("--image-column", default="patch_path", help="CSV column containing image paths.")
    parser.add_argument("--text-column", default="text", help="CSV column containing text descriptions.")
    parser.add_argument("--text-token-column", default=None, help="Optional CSV column with pre-tokenised text (unused).")
    parser.add_argument("--magnification-column", default="magnification", help="CSV column containing magnification levels.")
    parser.add_argument("--region-column", default="region_id", help="CSV column identifying spatial regions.")
    parser.add_argument("--strip-region-suffix", action="store_true", help="Strip magnification suffixes from region identifiers.")
    parser.add_argument("--location-x-column", default=None)
    parser.add_argument("--location-y-column", default=None)
    parser.add_argument("--location-width-column", default=None)
    parser.add_argument("--location-height-column", default=None)
    parser.add_argument("--location-size-column", default=None)
    parser.add_argument("--default-patch-size", type=float, default=None)
    parser.add_argument("--strict-files", action="store_true", help="Fail if an image path is missing on disk.")
    parser.add_argument("--filter-column", default=None, help="Optional column used to filter rows.")
    parser.add_argument("--filter-values", nargs="*", default=None, help="Explicit values to keep when filtering rows.")
    parser.add_argument(
        "--label-column",
        default=None,
        help="CSV column containing region labels to store alongside the extracted features.",
    )
    parser.add_argument(
        "--slide-column",
        default=None,
        help="CSV column identifying the slide for each region (used to disambiguate region IDs).",
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--model-name", default="musk_large_patch16_384", help="MUSK timm model identifier.")
    parser.add_argument("--checkpoint", default="hf_hub:xiangjx/musk", help="Checkpoint path or Hugging Face reference.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--ms-augment", action="store_true", help="Enable MUSK multi-scale augmentation during encoding.")
    parser.add_argument("--tokenizer-path", default=None, help="Path to tokenizer.spm (defaults to MUSK repo).")
    parser.add_argument("--text-max-length", type=int, default=100, help="Maximum token length passed to xlm_tokenizer.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated checkpoints.")
    parser.add_argument("--musk-repo", default=None, help="Path to a local MUSK clone (for imports and tokenizer).")
    parser.add_argument("--metadata-json", default=None, help="Optional path to dump metadata as JSON.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    _maybe_login(args.hf_token)

    tokenizer_path = args.tokenizer_path or _default_tokenizer_path(_resolve_repo_root(args.musk_repo))
    if tokenizer_path is None or not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(
            "Could not locate tokenizer.spm. Provide --tokenizer-path explicitly or --musk-repo pointing to"
            " a MUSK clone containing musk/models/tokenizer.spm."
        )

    encoder = MUSKEncoder(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        device=args.device,
        precision=args.precision,
        ms_augment=args.ms_augment,
        tokenizer_path=tokenizer_path,
        text_max_length=args.text_max_length,
        repo_root=args.musk_repo,
    )

    transform = _build_transform(image_size=384, antialias=True)

    dataset_kwargs = dict(
        image_root=args.image_root,
        region_column=args.region_column,
        magnification_column=args.magnification_column,
        image_column=args.image_column,
        text_column=args.text_column,
        text_token_column=args.text_token_column,
        location_x_column=args.location_x_column,
        location_y_column=args.location_y_column,
        location_width_column=args.location_width_column,
        location_height_column=args.location_height_column,
        location_size_column=args.location_size_column,
        default_patch_size=args.default_patch_size,
        strict_files=args.strict_files,
        strip_region_suffix=args.strip_region_suffix,
        filter_column=args.filter_column,
        filter_values=args.filter_values,
    )

    extract_musk_embeddings(
        csv_path=args.csv,
        output_path=args.output,
        encoder=encoder,
        image_transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_kwargs=dataset_kwargs,
        metadata_json=args.metadata_json,
        label_column=args.label_column,
        slide_column=args.slide_column,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

