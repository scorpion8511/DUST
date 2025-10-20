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
        tensors residing on ``device``.  The function assumes the first
        dimension enumerates samples and will reshape the embedding tensor if
        necessary so that ``embeddings.shape[0] == len(labels)``.
    """

    embeddings = _to_tensor(embeddings, device)
    labels = _to_tensor(labels, device).flatten()

    num_samples = labels.numel()
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(-1)

    if embeddings.shape[0] != num_samples:
        total_elements = embeddings.numel()
        if num_samples == 0 or total_elements % num_samples != 0:
            raise ValueError(
                "Embedding tensor cannot be reshaped to align with the provided labels."
            )
        feature_dim = total_elements // num_samples
        embeddings = embeddings.reshape(num_samples, feature_dim)

    classes = torch.unique(labels)
    class_histograms = {}

    flat_embeddings = embeddings.reshape(num_samples, -1)
    all_values = flat_embeddings.reshape(-1)
    min_val = all_values.min()
    max_val = all_values.max()
    bin_edges = torch.linspace(min_val, max_val, steps=num_bins + 1, device=device)

    for cls in classes:
        class_embeddings = flat_embeddings[labels == cls]
        if class_embeddings.numel() == 0:
            continue
        class_values = class_embeddings.reshape(-1)
        hist = torch.histogram(class_values, bins=bin_edges)[0].float()
        if hist.sum() > 0:
            hist /= hist.sum()
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
