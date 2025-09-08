"""MSCIV metric implementation."""
from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_tensor(x: ArrayLike) -> torch.Tensor:
    """Convert numpy arrays to tensors without copying when possible."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)


def msci_v(embeddings: ArrayLike) -> float:
    """Compute the Multi-Scale Cosine Invariance (MSCI-V) metric.

    Parameters
    ----------
    embeddings:
        Array of shape ``(n_regions, n_magnifications, dim)``.  The array
        should contain an embedding vector for every region at multiple
        magnifications.  The function accepts both :class:`numpy.ndarray` and
        :class:`torch.Tensor` inputs.

    Returns
    -------
    float
        Mean pairwise cosine similarity across magnifications and regions.
    """
    tensor = _to_tensor(np.asarray(embeddings))
    if tensor.ndim != 3:
        raise ValueError("`embeddings` must have shape (regions, magnifications, dim)")

    # Normalise each embedding vector.
    tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    # Compute pairwise cosine similarity between magnifications per region.
    # result shape: (regions, mags, mags)
    pairwise = torch.einsum("rmd,rnd->rmn", tensor, tensor)

    # Extract upper-triangular (i<j) values and average.
    mags = pairwise.shape[1]
    idx = torch.triu_indices(mags, mags, offset=1)
    scores = pairwise[:, idx[0], idx[1]]
    return scores.mean().item()


__all__ = ["msci_v"]
