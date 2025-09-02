import torch
from typing import Any


def _to_tensor(data: Any, device: str) -> torch.Tensor:
    """Convert ``data`` to a :class:`~torch.Tensor` on ``device``."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return torch.as_tensor(data, device=device)


def compute_histograms(embeddings: Any, labels: Any, num_bins: int = 50, device: str = "cpu"):
    """Compute class-wise histograms on the specified device.

    Parameters
    ----------
    embeddings, labels:
        Input data that can be a tensor or NumPy array. Both are converted to
        tensors residing on ``device``.
    """
    embeddings = _to_tensor(embeddings, device)
    labels = _to_tensor(labels, device)
    classes = torch.unique(labels)
    class_histograms = {}
    bin_edges = None
    for cls in classes:
        class_embeddings = embeddings[labels == cls]
        hist, bin_edges = torch.histogram(class_embeddings.flatten(), bins=num_bins, density=True)
        class_histograms[int(cls.item())] = hist
    return class_histograms, bin_edges


def histogram_intersection(hist1: torch.Tensor, hist2: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.minimum(hist1, hist2))


def compute_histogram_intersection_metric(
    embeddings: Any,
    labels: Any,
    num_bins: int = 50,
    device: str = "cpu",
) -> float:
    """Compute the Histogram of Intersections (HoI) metric using GPU tensors."""
    class_histograms, _ = compute_histograms(embeddings, labels, num_bins=num_bins, device=device)
    classes = list(class_histograms.keys())
    intersections = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            intersections.append(
                histogram_intersection(class_histograms[classes[i]], class_histograms[classes[j]])
            )
    avg_intersection = torch.stack(intersections).mean()
    confidence_score = 1.0 - avg_intersection
    return confidence_score.item()
