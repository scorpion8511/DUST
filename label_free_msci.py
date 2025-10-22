"""Compute MSCI for single-modality embeddings without labels.

This helper mirrors the MSCI implementation used by :mod:`scoring_pipeline`
but focuses on the common case where only multi-magnification image patches
are available.  The script accepts feature archives produced by either the
unimodal extractor (``extract_features.py``) or the multimodal image/text
pipelines.  When the input file does not contain an ``embeddings`` tensor the
loader collapses the per-magnification image embedding matrices into a single
``(num_regions * num_magnifications, dim)`` tensor so it can be processed by
``compute_msci_single_modality`` in label-free mode.

Example
=======

::

    python label_free_msci.py \
        /path/to/features.pth \
        --manifest /path/to/manifest.csv \
        --region-column patch_id \
        --magnification-column patch_scale \
        --magnifications 5 10 20 40

The manifest is optional when the feature archive already stores ``region_ids``
and ``magnifications`` arrays.  By default the same feature file is used for
both the "train" and "eval" payloads passed into
``compute_msci_single_modality``; provide ``--train-features`` when a separate
training split is available.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Mapping, Sequence

import torch

from scoring_pipeline import MSCIResult, compute_msci_single_modality


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _prepare_unimodal_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a mapping containing ``embeddings``/metadata for MSCI.

    ``compute_msci_single_modality`` expects a dictionary with an ``embeddings``
    tensor plus optional ``region_ids``/``magnifications`` entries.  Unimodal
    feature dumps produced by :mod:`extract_features` already follow that
    layout, so this helper simply normalises the tensor dtype and forwards the
    existing metadata.
    """

    if "embeddings" not in data:
        raise KeyError("Feature archive is missing an 'embeddings' tensor")

    payload: Dict[str, Any] = {"embeddings": _to_float_tensor(data["embeddings"])}

    for key in ("region_ids", "magnifications", "labels", "row_indices"):
        if key in data:
            payload[key] = data[key]

    return payload


def _prepare_multimodal_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten image embeddings from a multimodal feature dump for MSCI."""

    image_embeddings = data.get("image_embeddings")
    metadata = data.get("metadata")
    magnifications = data.get("magnifications")

    if not isinstance(image_embeddings, Mapping) or magnifications is None:
        raise ValueError("Multimodal feature archive is missing image embeddings")

    if not isinstance(metadata, Mapping) or "region_ids" not in metadata:
        raise ValueError("Multimodal metadata must contain a 'region_ids' list")

    region_ids: Sequence[str] = list(metadata["region_ids"])  # type: ignore[arg-type]
    if not region_ids:
        raise ValueError("No regions found in the multimodal metadata")

    embeddings_chunks: list[torch.Tensor] = []
    region_column: list[str] = []
    magnification_column: list[Any] = []

    for mag in magnifications:
        if mag not in image_embeddings:
            raise KeyError(f"Missing image embeddings for magnification {mag}")
        tensor = _to_float_tensor(image_embeddings[mag])
        if tensor.ndim != 2:
            raise ValueError(
                f"Magnification {mag} embeddings must be rank-2, received shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != len(region_ids):
            raise ValueError(
                f"Magnification {mag} contains {tensor.shape[0]} embeddings but metadata lists "
                f"{len(region_ids)} regions"
            )
        embeddings_chunks.append(tensor)
        region_column.extend(region_ids)
        magnification_column.extend([mag] * len(region_ids))

    combined_embeddings = torch.cat(embeddings_chunks, dim=0)

    payload: Dict[str, Any] = {
        "embeddings": combined_embeddings,
        "region_ids": region_column,
        "magnifications": magnification_column,
    }

    if "row_indices" in data:
        row_indices = data["row_indices"]
        if isinstance(row_indices, Sequence) and len(row_indices) == combined_embeddings.shape[0]:
            payload["row_indices"] = list(row_indices)

    return payload


def load_feature_payload(path: str) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")

    if "embeddings" in data:
        return _prepare_unimodal_payload(data)

    if "image_embeddings" in data:
        return _prepare_multimodal_payload(data)

    raise ValueError(
        "Unsupported feature archive format – expected 'embeddings' or 'image_embeddings' entries"
    )


def compute_label_free_msci(
    eval_features: Mapping[str, Any],
    *,
    train_features: Mapping[str, Any] | None = None,
    manifest: str | None = None,
    region_column: str = "region_id",
    magnification_column: str = "magnification",
    magnifications: Sequence[Any] | None = None,
) -> MSCIResult:
    train_payload = train_features or eval_features
    manifest_source = manifest if manifest else None

    return compute_msci_single_modality(
        train_payload,
        eval_features,
        manifest_source,
        region_column=region_column,
        magnification_column=magnification_column,
        label_column=None,
        magnifications=magnifications,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute MSCI for single-modality embeddings without labels"
    )
    parser.add_argument(
        "features",
        type=str,
        help="Path to the feature archive containing image embeddings",
    )
    parser.add_argument(
        "--train-features",
        type=str,
        default=None,
        help="Optional training feature archive; defaults to using the same file as eval",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional CSV manifest to align region/magnification metadata",
    )
    parser.add_argument(
        "--region-column",
        type=str,
        default="region_id",
        help="Manifest column (or feature metadata key) containing region identifiers",
    )
    parser.add_argument(
        "--magnification-column",
        type=str,
        default="magnification",
        help="Manifest column (or feature metadata key) containing magnification values",
    )
    parser.add_argument(
        "--magnifications",
        type=float,
        nargs="*",
        default=None,
        help="Optional subset of magnifications to evaluate",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to store the MSCI result as JSON",
    )
    return parser


def main() -> None:  # pragma: no cover - CLI entry point
    parser = build_argparser()
    args = parser.parse_args()

    eval_payload = load_feature_payload(args.features)
    train_payload = load_feature_payload(args.train_features) if args.train_features else None

    result = compute_label_free_msci(
        eval_payload,
        train_features=train_payload,
        manifest=args.manifest,
        region_column=args.region_column,
        magnification_column=args.magnification_column,
        magnifications=args.magnifications,
    )

    print("MSCI score (label-free): {:.6f}".format(result.score))
    print(
        "Mean variance: {:.6f} across {} / {} regions".format(
            result.mean_variance, result.regions_used, result.regions_total
        )
    )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "msci": result.score,
                    "mean_variance": result.mean_variance,
                    "max_variance": result.max_variance,
                    "regions_used": result.regions_used,
                    "regions_total": result.regions_total,
                    "per_region_variance": list(result.per_region_variance),
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

