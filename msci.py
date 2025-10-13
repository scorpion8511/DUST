"""Utilities for computing the Model Spatial Consistency Index (MSCI).

The MSCI score summarizes how consistent a model's confidences are within
predefined spatial regions (e.g., tiles originating from the same slide).
Lower intra-region variance indicates better spatial stability and therefore
translates into a higher MSCI score.  The helper below operates entirely on
PyTorch tensors so it can run on either CPU or GPU without additional data
movement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np
import torch


def _to_tensor(data: Any, device: torch.device) -> torch.Tensor:
    """Convert arbitrary region identifiers to a tensor on ``device``.

    The region identifiers may already be numeric or can be arbitrary hashable
    Python objects (e.g., strings).  In the latter case they are mapped to
    integer indices while preserving their relative ordering.
    """

    if isinstance(data, torch.Tensor):
        return data.to(device)

    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.number):
            return torch.as_tensor(data, device=device)
        data = data.tolist()

    if isinstance(data, (list, tuple)):
        if not data:
            return torch.empty(0, device=device, dtype=torch.long)
        first = data[0]
        if isinstance(first, (int, float, np.integer, np.floating)):
            return torch.as_tensor(data, device=device)
        # Map arbitrary identifiers (e.g., strings) to integers deterministically
        mapping = {value: idx for idx, value in enumerate(dict.fromkeys(data))}
        encoded = [mapping[value] for value in data]
        return torch.tensor(encoded, device=device)

    if isinstance(data, Iterable):
        materialized = list(data)
        return _to_tensor(materialized, device)

    raise TypeError(f"Unsupported region identifier type: {type(data)!r}")


@dataclass
class MSCIResult:
    """Container for MSCI statistics."""

    score: float
    mean_variance: float
    per_region_variances: List[float]
    regions_used: int
    total_regions: int


def compute_msci(
    logits: torch.Tensor,
    *,
    region_ids: Optional[Any] = None,
    device: str | torch.device = "cpu",
    min_region_size: int = 3,
    min_regions: int = 2,
) -> MSCIResult:
    """Compute the Model Spatial Consistency Index for a set of logits.

    Parameters
    ----------
    logits:
        Logits produced for each evaluation example.  The function operates on
        the confidence (maximum softmax probability) derived from these logits.
    region_ids:
        Identifiers that map each example to a spatial region.  Accepts tensors,
        numpy arrays, or any iterable of hashable Python objects.  When ``None``
        the MSCI score is undefined because no grouping information is
        available.
    device:
        Device on which computations should occur.
    min_region_size:
        Minimum number of samples required for a region to be considered.
    min_regions:
        Minimum number of qualifying regions required in order to report a
        finite MSCI score.
    """

    device = torch.device(device)
    logits = logits.to(device)
    n_samples = logits.shape[0]

    if region_ids is None:
        return MSCIResult(float("nan"), float("nan"), [], 0, 0)

    region_tensor = _to_tensor(region_ids, device)
    if region_tensor.ndim != 1 or region_tensor.numel() != n_samples:
        raise ValueError(
            "region_ids must be a one-dimensional collection matching the number of logits"
        )

    region_tensor = region_tensor.to(dtype=torch.long)
    unique_regions = torch.unique(region_tensor, sorted=False)
    total_regions = int(unique_regions.numel())

    probs = torch.softmax(logits, dim=-1)
    confidences = probs.max(dim=-1).values

    per_region_variances: List[float] = []
    regions_used = 0

    for region_id in unique_regions:
        mask = region_tensor == region_id
        count = int(mask.sum().item())
        if count < min_region_size:
            continue
        region_confidences = confidences[mask]
        if region_confidences.numel() < 2:
            continue
        variance = torch.var(region_confidences, unbiased=False).item()
        per_region_variances.append(float(variance))
        regions_used += 1

    if regions_used >= max(1, min_regions) and per_region_variances:
        mean_variance = float(np.mean(per_region_variances))
        score = float(max(0.0, 1.0 - mean_variance))
    else:
        mean_variance = float("nan")
        score = float("nan")

    return MSCIResult(
        score=score,
        mean_variance=mean_variance,
        per_region_variances=per_region_variances,
        regions_used=regions_used,
        total_regions=total_regions,
    )

