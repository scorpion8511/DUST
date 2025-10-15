"""Feature extraction pipeline for PLIP image-text encoders.

This script mirrors the interface of ``multimodal_feature_extraction.py`` but
relies on the `plip` library to obtain embeddings from the public
``vinid/plip`` checkpoint (or a user-specified variant). It consumes a CSV file
that enumerates paired image/text examples – optionally including patch
locations and magnification metadata – and produces a ``.pth`` archive
compatible with ``multimodal_scoring.py``.

Example
-------

.. code-block:: bash

    python plip_feature_extraction.py dataset.csv plip_features.pth \
        --image-root /path/to/images --image-column patch_path \
        --text-column generated_text --magnification-column patch_scale \
        --region-column patch_id --strip-region-suffix

The resulting ``plip_features.pth`` file contains ``image_embeddings`` (grouped
by magnification), ``text_embeddings`` aligned with each region, and a
``metadata`` dictionary mirroring the TorchScript extractor to streamline
scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multimodal_feature_extraction import ImageTextCSVDataset


@dataclass
class RegionAccumulator:
    """Stores embeddings and metadata for a spatial region."""

    region_id: str
    text_sum: Optional[torch.Tensor] = None
    text_count: int = 0
    text_values: List[str] = field(default_factory=list)
    images: Dict[int, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, object] = field(
        default_factory=lambda: {"patches": {}, "samples": [], "csv_region_ids": set()}
    )


def build_collate_fn():
    """Create a collate function that preserves PIL images and metadata."""

    def collate(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        images = [item["image"] for item in batch]
        texts = [item.get("text") for item in batch]
        if any(text is None for text in texts):
            raise ValueError(
                "Encountered missing text entries; ensure --text-column is present in the CSV."
            )

        magnifications = [int(item["magnification"]) for item in batch]
        region_ids = [str(item["region_id"]) for item in batch]
        metadata = [item["metadata"] for item in batch]

        return {
            "images": images,
            "texts": [text or "" for text in texts],
            "magnifications": magnifications,
            "region_ids": region_ids,
            "metadata": metadata,
        }

    return collate


def extract_plip_embeddings(
    dataloader: DataLoader,
    model: "PLIP",
    plip_batch_size: int,
    normalize: bool,
    magnifications: Optional[Sequence[int]],
    drop_missing: bool,
) -> Dict[str, object]:
    regions: "OrderedDict[str, RegionAccumulator]" = OrderedDict()
    observed_magnifications: set[int] = set()

    for batch in dataloader:
        images = batch["images"]
        texts = batch["texts"]

        image_embeddings = model.encode_images(images, batch_size=plip_batch_size)
        text_embeddings = model.encode_text(texts, batch_size=plip_batch_size)

        image_embeddings = torch.from_numpy(np.asarray(image_embeddings)).float()
        text_embeddings = torch.from_numpy(np.asarray(text_embeddings)).float()

        if normalize:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

        for idx, region_id in enumerate(batch["region_ids"]):
            magnification = int(batch["magnifications"][idx])
            observed_magnifications.add(magnification)
            accumulator = regions.setdefault(region_id, RegionAccumulator(region_id=region_id))

            embedding_image = image_embeddings[idx].cpu()
            embedding_text = text_embeddings[idx].cpu()

            current_text = texts[idx]
            if accumulator.text_sum is None:
                accumulator.text_sum = embedding_text.clone()
            else:
                accumulator.text_sum = accumulator.text_sum + embedding_text
            accumulator.text_count += 1
            accumulator.text_values.append(current_text)

            accumulator.images[magnification] = embedding_image
            accumulator.metadata.setdefault("samples", []).append(batch["metadata"][idx])
            patch = batch["metadata"][idx].get("patch")
            if patch is not None:
                accumulator.metadata.setdefault("patches", {})[magnification] = patch
            original_id = batch["metadata"][idx].get("csv_region_id", region_id)
            accumulator.metadata.setdefault("csv_region_ids", set()).add(original_id)

    if magnifications is None:
        ordered_magnifications = sorted(observed_magnifications)
    else:
        ordered_magnifications = [int(m) for m in magnifications]
        missing = set(ordered_magnifications) - observed_magnifications
        if missing:
            raise ValueError(
                f"Requested magnifications {sorted(missing)} were not found in the dataset."
            )

    region_ids: List[str] = []
    text_embeddings: List[torch.Tensor] = []
    raw_texts: List[Optional[str]] = []
    csv_region_mapping: Dict[str, List[str]] = {}
    patches = {mag: [] for mag in ordered_magnifications}

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
            if len(unique_texts) == 1:
                raw_texts.append(unique_texts[0])
            else:
                raw_texts.append(unique_texts)
        else:
            raw_texts.append(None)

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

        # When drop_missing removes a region, undo partial writes.
        region_ids.pop()
        text_embeddings.pop()
        raw_texts.pop()
        csv_region_mapping.pop(region_id, None)
        for mag in ordered_magnifications:
            if patches[mag]:
                patches[mag].pop()

    if not region_ids:
        raise ValueError(
            "All regions were filtered out – disable --drop-missing or verify magnification coverage."
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
        image_tensors[mag] = torch.stack(embeddings, dim=0)  # type: ignore[arg-type]

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
        with open(metadata_json, "w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract PLIP features from a CSV manifest.")
    parser.add_argument("csv", type=str, help="Path to the CSV file with image/text pairs.")
    parser.add_argument("output", type=str, help="Destination .pth file for the extracted features.")

    parser.add_argument(
        "--model",
        type=str,
        default="vinid/plip",
        help="Identifier or path for the PLIP checkpoint (default: vinid/plip).",
    )
    parser.add_argument("--image-root", type=str, default=None, help="Optional root to prepend to relative image paths.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of samples per DataLoader batch.")
    parser.add_argument(
        "--plip-batch-size",
        type=int,
        default=32,
        help="Batch size forwarded to PLIP encode methods (defaults to loader batch size).",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable L2 normalisation for image and text embeddings (enabled by default).",
    )
    parser.set_defaults(normalize=True)

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
        help="Fallback square size when only patch coordinates are supplied.",
    )
    parser.add_argument("--strict-files", action="store_true", help="Raise if image files referenced in the CSV are missing.")
    parser.add_argument(
        "--strip-region-suffix",
        action="store_true",
        help="Remove trailing magnification tokens (e.g. 'patch_0_5x') when grouping regions.",
    )
    parser.add_argument("--filter-column", type=str, default=None)
    parser.add_argument(
        "--filter-values",
        type=str,
        nargs="*",
        default=None,
        help="Keep rows whose filter-column value matches one of the provided tokens.",
    )
    parser.add_argument(
        "--magnifications",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of magnifications to retain; defaults to those found in the CSV.",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Discard regions lacking at least one of the requested magnifications instead of failing.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="Optional path to write the metadata dictionary as JSON alongside the .pth archive.",
    )

    return parser


def run(args: argparse.Namespace) -> Dict[str, object]:
    from plip.plip import PLIP

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

    collate_fn = build_collate_fn()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = PLIP(args.model)
    plip_batch_size = args.plip_batch_size or args.batch_size

    features = extract_plip_embeddings(
        dataloader=loader,
        model=model,
        plip_batch_size=plip_batch_size,
        normalize=args.normalize,
        magnifications=args.magnifications,
        drop_missing=args.drop_missing,
    )

    metadata = features.get("metadata")
    if isinstance(metadata, dict):
        metadata["plip_model"] = args.model

    save_outputs(features, args.output, args.metadata_json)
    return features


def main() -> None:  # pragma: no cover - CLI entry point
    parser = build_argparser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
