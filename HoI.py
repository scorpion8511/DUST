import numpy as np
import torch
from typing import Dict, Tuple


def to_numpy(data):
    """Convert tensors to CPU NumPy arrays."""
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    raise ValueError("Unsupported data type: %r" % type(data))


class LDA:
    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage

    def _cov(self, X: np.ndarray) -> np.ndarray:
        emp_cov = np.cov(X.T, bias=1)
        n_features = emp_cov.shape[0]
        mu = np.trace(emp_cov) / n_features
        shrunk_cov = (1.0 - self.shrinkage) * emp_cov
        shrunk_cov.flat[:: n_features + 1] += self.shrinkage * mu
        return shrunk_cov

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDA":
        classes, y_t = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(y_t) / float(len(y))
        means = np.zeros((len(classes), X.shape[1]))
        np.add.at(means, y_t, X)
        means /= np.bincount(y_t)[:, None]
        self.means_ = means

        Sw = np.zeros((X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y_t == idx]
            Sw += self.priors_[idx] * self._cov(Xg)
        St = self._cov(X)
        Sb = St - Sw

        evals, evecs = np.linalg.eigh(np.linalg.pinv(Sw).dot(Sb))
        evecs = evecs[:, np.argsort(evals)[::-1]]
        self.coef_ = means @ evecs @ evecs.T
        self.intercept_ = -0.5 * np.diag(means @ self.coef_.T) + np.log(self.priors_)
        return self


def compute_histograms(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 50,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    bin_edges = np.histogram(embeddings.flatten(), bins=num_bins, density=True)[1]
    class_histograms: Dict[int, np.ndarray] = {}
    for cls in np.unique(labels):
        class_emb = embeddings[labels == cls]
        hist, _ = np.histogram(class_emb.flatten(), bins=bin_edges, density=True)
        class_histograms[int(cls)] = hist
    return class_histograms, bin_edges


def histogram_intersection(hist1: np.ndarray, hist2: np.ndarray) -> float:
    return np.sum(np.minimum(hist1, hist2))


def compute_histogram_intersection_metric(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 50,
) -> float:
    class_histograms, _ = compute_histograms(embeddings, labels, num_bins=num_bins)
    classes = list(class_histograms.keys())
    scores = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            scores.append(
                histogram_intersection(
                    class_histograms[classes[i]], class_histograms[classes[j]]
                )
            )
    average_intersection = np.mean(scores)
    return 1.0 - average_intersection


def compute_scores_for_all_models(
    model_paths: Dict[str, Tuple[str, str]]
) -> None:
    for model_name, (train_path, eval_path) in model_paths.items():
        print(f"Processing model: {model_name}")
        try:
            train = torch.load(train_path)
            evald = torch.load(eval_path)
            X_train = to_numpy(train["embeddings"])
            y_train = to_numpy(train["labels"])
            X_eval = to_numpy(evald["embeddings"])
            y_eval = to_numpy(evald["labels"])

            lda = LDA(shrinkage=0.1)
            lda.fit(X_train, y_train)
            embeddings = X_eval @ lda.coef_.T + lda.intercept_
            score = compute_histogram_intersection_metric(
                embeddings, y_eval, num_bins=50
            )
            print(
                f"Histogram Intersection Confidence Score for {model_name}: {score}"
            )
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

