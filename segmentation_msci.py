r"""Compute the Magnification-Scale Consistency Index (MSCI) for segmentation.

This utility mirrors the classification-centric MSCI routine in
``scoring_pipeline.py`` but operates directly on dense segmentation outputs.
For each region ``r`` and magnification ``m`` we assume a per-pixel
probability tensor ``P_{r,m} \in [0, 1]^{C \times H \times W}``.  The script
first averages the probabilities across spatial locations to obtain a
class-distribution descriptor ``\bar{p}_{r,m}`` which is normalised to unit
length.  Let ``M_r`` denote the set of magnifications retained for the region
after optional filtering.  The region prototype is the normalised centroid

.. math::

   c_r = \frac{\sum_{m \in M_r} \bar{p}_{r,m}}{\lVert \sum_{m \in M_r} \bar{p}_{r,m} \rVert_2}.

We quantify magnification agreement via cosine similarities

.. math::

   s_{r,m} = \langle \bar{p}_{r,m}, c_r \rangle,

whose empirical variance ``\sigma_r^2`` is computed over ``M_r``.  Following the
original MSCI definition, each variance is normalised by ``\sigma^2_{\max}``, the
maximum attainable variance for ``|M_r|`` magnifications, before averaging over
regions:

.. math::

   \text{MSCI} = 1 - \frac{1}{|R|} \sum_{r \in R} \frac{\sigma_r^2}{\sigma^2_{\max}(|M_r|)}.

The normalisation constant matches ``scoring_pipeline._max_variance`` and clips
the final score to ``[0, 1]``.  Regions that expose fewer than two magnification
levels are ignored because their variance is undefined.

Usage
-----

```
python segmentation_msci.py \
    /path/to/segmentation_predictions.pth \
    --manifest /path/to/metadata.csv \
    --region-column region_id \
    --magnification-column magnification \
    --magnifications 5 10 20 40
```

The feature archive must contain either a ``probabilities`` tensor of shape
``[N, C, H, W]`` or a ``logits`` tensor with the same dimensions (in which case a
softmax is applied).  Metadata can be sourced from the archive itself when
``region_ids`` and ``magnifications`` arrays are present; otherwise provide a CSV
manifest aligned with the prediction tensor.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scoring_pipeline import MSCIResult, _max_variance


def _as_float_tensor(value: Any) -> torch.Tensor:
    """Convert ``value`` to a detached ``float32`` tensor on the CPU."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _parse_magnification(value: Any) -> int:
    """Coerce magnification strings such as ``"20x"`` to integers."""

    text = str(value).strip().lower().replace("×", "x")
    if text.endswith("x"):
        text = text[:-1]

    try:
        numeric = float(text)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unable to interpret magnification value '{value}'") from exc

    return int(round(numeric))


def _load_probabilities(
    features: Mapping[str, Any], probability_key: str, logits_key: str | None
) -> torch.Tensor:
    """Extract a ``[N, C, H, W]`` probability tensor from ``features``."""

    if probability_key in features:
        tensor = _as_float_tensor(features[probability_key])
    elif logits_key and logits_key in features:
        logits = _as_float_tensor(features[logits_key])
        tensor = F.softmax(logits, dim=1)
    else:
        available = ", ".join(sorted(features.keys()))
        raise KeyError(
            f"Neither '{probability_key}' nor '{logits_key}' found in feature archive. "
            f"Available keys: {available}"
        )

    if tensor.ndim != 4:
        raise ValueError(
            f"Segmentation predictions must have shape [N, C, H, W]; received {tuple(tensor.shape)}"
        )

    return tensor


def _resolve_metadata(
    features: Mapping[str, Any],
    manifest: pd.DataFrame | None,
    region_column: str,
    magnification_column: str,
    expected: int,
) -> tuple[list[str], list[int]]:
    """Return aligned region identifiers and magnifications."""

    if manifest is not None:
        if region_column not in manifest.columns:
            raise KeyError(f"Manifest is missing region column '{region_column}'")
        if magnification_column not in manifest.columns:
            raise KeyError(
                f"Manifest is missing magnification column '{magnification_column}'"
            )
        regions = [str(v) for v in manifest[region_column].tolist()]
        magnifications = manifest[magnification_column].tolist()
        if len(regions) != expected:
            raise ValueError(
                "Manifest row count does not match the number of segmentation predictions"
            )
        return regions, [_parse_magnification(m) for m in magnifications]

    region_values = features.get("region_ids")
    magnification_values = features.get("magnifications")

    if region_values is None or magnification_values is None:
        raise ValueError(
            "Segmentation archive must include 'region_ids' and 'magnifications' or an external manifest"
        )

    regions = [str(v) for v in region_values]
    magnifications = list(magnification_values)

    if len(regions) != expected or len(magnifications) != expected:
        raise ValueError("Feature metadata lengths do not align with the predictions tensor")

    return regions, [_parse_magnification(m) for m in magnifications]


def _aggregate_probability(prob_map: torch.Tensor) -> torch.Tensor | None:
    """Average class probabilities over spatial locations and normalise."""

    c, h, w = prob_map.shape
    if h == 0 or w == 0:
        return None

    flattened = prob_map.view(c, -1)
    mean_vec = flattened.mean(dim=1)
    norm = torch.norm(mean_vec)
    if torch.isnan(norm) or float(norm.item()) == 0.0:
        return None
    return mean_vec / norm


def compute_segmentation_msci(
    probabilities: torch.Tensor,
    regions: Sequence[str],
    magnifications: Sequence[int],
    *,
    requested_magnifications: Iterable[int] | None = None,
) -> MSCIResult:
    """Return the MSCI score for segmentation predictions across magnifications."""

    if probabilities.ndim != 4:
        raise ValueError("Probabilities must be a [N, C, H, W] tensor")
    if probabilities.shape[0] != len(regions) or probabilities.shape[0] != len(magnifications):
        raise ValueError("Predictions, regions, and magnifications must share the same length")

    requested = (
        {_parse_magnification(m) for m in requested_magnifications}
        if requested_magnifications
        else None
    )

    region_vectors: Dict[str, Dict[int, list[torch.Tensor]]] = {}

    for idx in range(probabilities.shape[0]):
        region = regions[idx]
        mag = int(magnifications[idx])
        if requested and mag not in requested:
            continue

        vec = _aggregate_probability(probabilities[idx])
        if vec is None:
            continue

        region_entry = region_vectors.setdefault(region, {})
        region_entry.setdefault(mag, []).append(vec)

    per_region_variances: list[float] = []
    per_region_max_variances: list[float] = []

    for region, mag_vectors in region_vectors.items():
        available_mags = sorted(mag_vectors)
        if len(available_mags) < 2:
            continue

        aggregated: list[torch.Tensor] = []
        for mag in available_mags:
            vectors = mag_vectors[mag]
            stacked = torch.stack(vectors, dim=0)
            mean_vec = stacked.mean(dim=0)
            norm = torch.norm(mean_vec)
            if torch.isnan(norm) or float(norm.item()) == 0.0:
                continue
            aggregated.append(mean_vec / norm)

        if len(aggregated) < 2:
            continue

        centroid = torch.stack(aggregated, dim=0).mean(dim=0)
        centroid_norm = torch.norm(centroid)
        if torch.isnan(centroid_norm) or float(centroid_norm.item()) == 0.0:
            continue
        centroid = centroid / centroid_norm

        sims = torch.stack([torch.dot(vec, centroid) for vec in aggregated], dim=0)
        mean_sim = sims.mean()
        variance = torch.mean((sims - mean_sim) ** 2)

        var_value = float(variance.item())
        per_region_variances.append(var_value)
        per_region_max_variances.append(_max_variance(len(aggregated)))

    regions_used = len(per_region_variances)
    regions_total = len(region_vectors)

    if regions_used == 0:
        raise ValueError(
            "No regions retained at the required magnifications; cannot compute segmentation MSCI"
        )

    variance_tensor = torch.tensor(per_region_variances, dtype=torch.float32)
    mean_variance = float(variance_tensor.mean().item())

    normalised = []
    for value, max_var in zip(per_region_variances, per_region_max_variances):
        if max_var > 0:
            normalised.append(value / max_var)
    normalised_tensor = torch.tensor(normalised, dtype=torch.float32) if normalised else torch.tensor(0.0)
    mean_normalised_variance = float(normalised_tensor.mean().item()) if normalised else 0.0

    score = max(0.0, min(1.0, 1.0 - mean_normalised_variance))
    representative_max = float(np.mean(per_region_max_variances)) if per_region_max_variances else 0.0

    return MSCIResult(
        score=score,
        mean_variance=mean_variance,
        max_variance=representative_max,
        per_region_variance=per_region_variances,
        regions_used=regions_used,
        regions_total=regions_total,
    )


def _load_manifest(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    manifest = pd.read_csv(path)
    if manifest.empty:
        raise ValueError("Manifest is empty; cannot align segmentation predictions")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the magnification-scale consistency index for segmentation outputs."
    )
    parser.add_argument(
        "features",
        help="Path to a torch archive containing segmentation probabilities or logits.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional CSV manifest aligned with the predictions tensor.",
    )
    parser.add_argument(
        "--region-column",
        default="region_id",
        help="Column name providing region identifiers when a manifest is supplied.",
    )
    parser.add_argument(
        "--magnification-column",
        default="magnification",
        help="Column name providing magnification levels when a manifest is supplied.",
    )
    parser.add_argument(
        "--magnifications",
        nargs="*",
        type=int,
        help="Optional subset of magnifications to include in the MSCI computation.",
    )
    parser.add_argument(
        "--probability-key",
        default="probabilities",
        help="Key inside the feature archive that stores per-pixel class probabilities.",
    )
    parser.add_argument(
        "--logits-key",
        default="logits",
        help="Fallback key for raw logits when probabilities are unavailable.",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to store the MSCI summary as JSON.",
    )

    args = parser.parse_args()

    features: Mapping[str, Any] = torch.load(args.features, map_location="cpu")
    manifest = _load_manifest(args.manifest)

    probabilities = _load_probabilities(features, args.probability_key, args.logits_key)
    regions, magnifications = _resolve_metadata(
        features,
        manifest,
        args.region_column,
        args.magnification_column,
        expected=probabilities.shape[0],
    )

    result = compute_segmentation_msci(
        probabilities,
        regions,
        magnifications,
        requested_magnifications=args.magnifications,
    )

    print("Segmentation MSCI score: {:.6f}".format(result.score))
    print("Mean variance across regions: {:.6f}".format(result.mean_variance))
    print("Average max variance: {:.6f}".format(result.max_variance))
    print("Regions used / total: {} / {}".format(result.regions_used, result.regions_total))

    if args.summary_json:
        pd.Series(asdict(result)).to_json(args.summary_json, indent=2)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

