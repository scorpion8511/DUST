import math
import time
from typing import List

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


def Energy_Score(logits: torch.Tensor, percent: int, tail: str, device: str = "cpu") -> float:
    logits = to_tensor(logits, device)
    energy = torch.logsumexp(logits, dim=-1)
    k = int(percent * len(energy) // 100)
    chs = torch.argsort(energy)
    chs = chs[:k] if tail == "bot" else chs[-k:]
    return energy[chs].mean().item()


def pad_to_square(tensor: torch.Tensor) -> torch.Tensor:
    size = tensor.numel()
    next_square = int(math.ceil(math.sqrt(size)) ** 2)
    padded = torch.zeros(next_square, device=tensor.device, dtype=tensor.dtype)
    padded[:size] = tensor
    return padded


def gabor_kernel(frequency: float, kernel_size: int = 31, sigma: float = 4.0, device: str = "cpu") -> torch.Tensor:
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    x = torch.linspace(-xmax, xmax, steps=kernel_size, device=device)
    y = torch.linspace(-ymax, ymax, steps=kernel_size, device=device)
    y, x = torch.meshgrid(y, x, indexing="ij")
    rotx = x
    g = torch.exp(-0.5 * (rotx**2 + y**2) / sigma**2)
    g *= torch.cos(2 * math.pi * frequency * rotx)
    return g


def compute_gabor_features(features: torch.Tensor, frequencies: List[float] | None = None, device: str = "cpu") -> torch.Tensor:
    if frequencies is None:
        frequencies = [0.1, 0.2, 0.3]
    features = to_tensor(features, device)
    gabor_feats = []
    for feature in features:
        padded_feature = pad_to_square(feature)
        dim = int(math.sqrt(padded_feature.numel()))
        image = padded_feature.view(1, 1, dim, dim)
        image_feats = []
        for freq in frequencies:
            kernel = gabor_kernel(freq, device=device).unsqueeze(0).unsqueeze(0)
            resp = F.conv2d(image, kernel, padding=kernel.shape[-1] // 2)
            image_feats.append(resp.flatten())
        gabor_feats.append(torch.cat(image_feats))
    return torch.stack(gabor_feats)


def compute_gabor_scores(embeddings: torch.Tensor, labels: torch.Tensor, frequencies: List[float] | None = None, device: str = "cpu") -> dict:
    embeddings = to_tensor(embeddings, device)
    labels = to_tensor(labels, device)
    gabor_features = compute_gabor_features(embeddings, frequencies=frequencies, device=device)
    lda = LDA(shrinkage=0.1, device=device)
    lda.fit(gabor_features, labels)
    probs = lda.predict_proba(gabor_features)
    lda_score = probs[torch.arange(len(labels), device=device), labels].mean()
    logits = gabor_features @ lda.coef_.T + lda.intercept_
    energy_score = Energy_Score(logits, percent=100, tail="bot", device=device)
    return {"lda_score": lda_score.item(), "energy_score": energy_score}


def benchmark_runtime(n_samples: int = 64, feature_dim: int = 512) -> dict:
    timings = {}
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; skipping GPU benchmark.")
            continue
        features = torch.randn(n_samples, feature_dim, device=device)
        labels = torch.randint(0, 2, (n_samples,), device=device)
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        compute_gabor_scores(features, labels, device=device)
        torch.cuda.synchronize() if device == "cuda" else None
        timings[device] = time.time() - start
    return timings
