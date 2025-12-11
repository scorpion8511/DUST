import math
import time
from typing import List, Optional

import torch
import torch.nn.functional as F


def to_tensor(data, device):
    """Convert input data to a torch.Tensor on the given device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return torch.tensor(data, device=device)


class LDA:
    """Simple Linear Discriminant Analysis implemented with PyTorch."""

    def __init__(self, shrinkage: float | None = None, priors=None, n_components=None, device: str = "cpu"):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.device = device

    def _cov(self, X: torch.Tensor, shrinkage: float = -1) -> torch.Tensor:
        X = X - X.mean(0)
        emp_cov = X.T @ X / (X.shape[0] - 1)
        if shrinkage < 0:
            return emp_cov
        n_features = emp_cov.shape[0]
        mu = torch.trace(emp_cov) / n_features
        shrunk_cov = (1.0 - shrinkage) * emp_cov
        idx = torch.arange(n_features, device=self.device)
        shrunk_cov[idx, idx] += shrinkage * mu
        return shrunk_cov

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X, y = to_tensor(X, self.device), to_tensor(y, self.device)
        self.classes_, y = torch.unique(y, return_inverse=True)
        cnt = torch.bincount(y)
        self.priors_ = cnt.float() / len(y)

        means = torch.zeros((len(self.classes_), X.shape[1]), device=self.device)
        means.index_add_(0, y, X)
        means /= cnt.unsqueeze(1)
        self.means_ = means

        cov = torch.zeros((X.shape[1], X.shape[1]), device=self.device)
        for idx, _ in enumerate(self.classes_):
            Xg = X[y == idx]
            cov += self.priors_[idx] * self._cov(Xg)
        self.covariance_ = cov

        Sw = self.covariance_
        shrink = 0.1 if self.shrinkage is None else self.shrinkage
        St = self._cov(X, shrinkage=shrink)
        n_features = Sw.shape[0]
        mu = torch.trace(Sw) / n_features
        shrunk_Sw = (1.0 - shrink) * Sw
        idx = torch.arange(n_features, device=self.device)
        shrunk_Sw[idx, idx] += shrink * mu
        Sb = St - shrunk_Sw

        inv_sw = torch.linalg.inv(shrunk_Sw)
        mat = inv_sw @ Sb
        evals, evecs = torch.linalg.eig(mat)
        evals = evals.real
        evecs = evecs.real
        idx = torch.argsort(evals, descending=True)
        evecs = evecs[:, idx]

        self.scalings_ = evecs
        self.coef_ = self.means_ @ evecs @ evecs.T
        self.intercept_ = -0.5 * torch.diagonal(self.means_ @ self.coef_.T) + torch.log(self.priors_)
        return self

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        X = to_tensor(X, self.device)
        logits = X @ self.coef_.T + self.intercept_
        return torch.softmax(logits, dim=1)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X = to_tensor(X, self.device)
        return X @ self.scalings_


def Energy_Score(logits: torch.Tensor, percent: int, tail: str, device: str = "cpu") -> float:
    logits = to_tensor(logits, device)
    energy = torch.logsumexp(logits, dim=-1)
    k = int(percent * len(energy) // 100)
    chs = torch.argsort(energy)
    chs = chs[:k] if tail == "bot" else chs[-k:]
    return energy[chs].mean().item()


def pad_to_square_batch(tensors: torch.Tensor) -> torch.Tensor:
    """Pad a batch of 1-D features to form square images."""
    n, size = tensors.shape
    next_square = int(math.ceil(math.sqrt(size)) ** 2)
    if next_square != size:
        pad = torch.zeros(n, next_square - size, device=tensors.device, dtype=tensors.dtype)
        tensors = torch.cat([tensors, pad], dim=1)
    dim = int(math.sqrt(next_square))
    return tensors.view(n, 1, dim, dim)


def gabor_filter_bank(
    frequencies: List[float],
    orientations: List[float],
    kernel_size: int = 31,
    sigma: float = 4.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Create a bank of real/imaginary Gabor kernels stacked for conv2d."""
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    x = torch.linspace(-xmax, xmax, steps=kernel_size, device=device)
    y = torch.linspace(-ymax, ymax, steps=kernel_size, device=device)
    y, x = torch.meshgrid(y, x, indexing="ij")
    kernels = []
    for freq in frequencies:
        for theta in orientations:
            rotx = x * math.cos(theta) + y * math.sin(theta)
            roty = -x * math.sin(theta) + y * math.cos(theta)
            g = torch.exp(-0.5 * (rotx**2 + roty**2) / sigma**2)
            kernels.append(g * torch.cos(2 * math.pi * freq * rotx))
            kernels.append(g * torch.sin(2 * math.pi * freq * rotx))
    return torch.stack(kernels).unsqueeze(1)


def compute_gabor_features(
    features: torch.Tensor,
    frequencies: Optional[List[float]] = None,
    orientations: Optional[List[float]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    if frequencies is None:
        frequencies = [0.1, 0.2, 0.3]
    if orientations is None:
        orientations = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    features = to_tensor(features, device)
    images = pad_to_square_batch(features)
    kernels = gabor_filter_bank(frequencies, orientations, device=device)
    k = kernels.shape[-1] // 2
    resp = F.conv2d(images, kernels, padding=k)
    real = resp[:, 0::2]
    imag = resp[:, 1::2]
    magnitude = torch.sqrt(real**2 + imag**2)
    return magnitude.reshape(magnitude.shape[0], -1)


def _apply_pca(train: torch.Tensor, evald: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor]:
    train = train - train.mean(0, keepdim=True)
    evald = evald - train.mean(0, keepdim=True)
    q = min(n_components, train.shape[0], train.shape[1])
    U, S, V = torch.pca_lowrank(train, q=q)
    W = V[:, :q]
    return train @ W, evald @ W


def compute_gabor_scores(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    eval_embeddings: torch.Tensor,
    eval_labels: torch.Tensor,
    frequencies: Optional[List[float]] = None,
    orientations: Optional[List[float]] = None,
    pca_dim: Optional[int] = 128,
    device: str = "cpu",
) -> dict:
    """Compute training-free Gabor energy and Fisher scores.

    The Gabor responses are standardised (subtract the train-set mean and divide
    by the train-set standard deviation) before PCA/LDA so that the downstream
    classifier and log-sum-exp energy are driven by relative texture contrast
    rather than absolute response magnitude. This normalisation makes energy
    scores comparable across models and magnifications, letting the final
    log-sum-exp aggregate reflect how discriminative the filtered features are
    instead of how large the raw activations happen to be.
    """
    train_embeddings = to_tensor(train_embeddings, device)
    eval_embeddings = to_tensor(eval_embeddings, device)
    train_labels = to_tensor(train_labels, device).long()
    eval_labels = to_tensor(eval_labels, device).long()

    train_feats = compute_gabor_features(
        train_embeddings, frequencies=frequencies, orientations=orientations, device=device
    )
    eval_feats = compute_gabor_features(
        eval_embeddings, frequencies=frequencies, orientations=orientations, device=device
    )

    mean = train_feats.mean(0, keepdim=True)
    std = train_feats.std(0, keepdim=True).clamp_min(1e-6)
    train_feats = (train_feats - mean) / std
    eval_feats = (eval_feats - mean) / std

    if pca_dim is not None:
        pca_dim = min(pca_dim, train_feats.shape[0], train_feats.shape[1])
        if pca_dim > 0:
            train_feats, eval_feats = _apply_pca(train_feats, eval_feats, pca_dim)

    lda = LDA(shrinkage=0.1, device=device)
    lda.fit(train_feats, train_labels)
    logits = eval_feats @ lda.coef_.T + lda.intercept_
    energy_score = Energy_Score(logits, percent=100, tail="bot", device=device)
    fisher = fisher_score(eval_feats, eval_labels)
    return {
        "energy": energy_score,
        "fisher": fisher,
        "combined": energy_score + fisher,
    }


def fisher_score(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute Fisher discriminant ratio for class separation."""
    labels = labels.long()
    classes, counts = torch.unique(labels, return_counts=True)
    overall = features.mean(0)
    sb = torch.zeros(1, device=features.device)
    sw = torch.zeros(1, device=features.device)
    for cls, cnt in zip(classes, counts):
        feats = features[labels == cls]
        mean = feats.mean(0)
        sb += cnt * torch.sum((mean - overall) ** 2)
        sw += torch.sum((feats - mean) ** 2)
    return (sb / (sw + 1e-6)).item()


def benchmark_runtime(n_samples: int = 64, feature_dim: int = 512) -> dict:
    """Benchmark scoring runtime on CPU vs. GPU.

    Parameters
    ----------
    n_samples: int
        Number of synthetic samples used for the benchmark.
    feature_dim: int
        Dimensionality of each synthetic embedding.

    Returns
    -------
    dict
        Mapping of device name to elapsed time in seconds.
    """

    timings = {}
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; skipping GPU benchmark.")
            continue

        features = torch.randn(n_samples, feature_dim, device=device)
        labels = torch.randint(0, 2, (n_samples,), device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        compute_gabor_scores(features, labels, features, labels, device=device)
        if device == "cuda":
            torch.cuda.synchronize()

        timings[device] = time.time() - start

    if {"cpu", "cuda"}.issubset(timings.keys()):
        speedup = timings["cpu"] / timings["cuda"]
        timings["speedup"] = speedup
    return timings
