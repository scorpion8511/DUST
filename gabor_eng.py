import numpy as np
import torch
from skimage.filters import gabor
from typing import Iterable, Sequence


def to_numpy(data):
    """Convert tensors to NumPy arrays."""
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.numpy()
    raise ValueError("Unsupported data type: %r" % type(data))


class LDA:
    """Simple LDA with optional covariance shrinkage."""

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
        self.classes_ = classes

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
        self.scalings_ = evecs
        self.coef_ = means @ evecs @ evecs.T
        self.intercept_ = -0.5 * np.diag(means @ self.coef_.T) + np.log(self.priors_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.coef_.T + self.intercept_
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def Energy_Score(logits: Sequence[np.ndarray], percent: int = 100, tail: str = "bot") -> float:
    """Compute mean energy score for selected percentile of samples."""
    logits = torch.as_tensor(logits)
    energy = torch.logsumexp(logits, dim=-1).numpy()
    k = max(1, int(percent * len(energy) / 100))
    idx = np.argsort(energy)
    chosen = idx[:k] if tail == "bot" else idx[-k:]
    return energy[chosen].mean()


def pad_to_square(array: np.ndarray) -> np.ndarray:
    size = array.size
    next_square = int(np.ceil(np.sqrt(size)) ** 2)
    padded = np.zeros(next_square, dtype=array.dtype)
    padded[:size] = array
    return padded


def compute_gabor_features(
    features: np.ndarray,
    frequencies: Iterable[float] = (0.1, 0.2, 0.3),
    orientations: Iterable[float] = (0.0,),
) -> np.ndarray:
    """Apply Gabor filters and return standardized magnitude responses."""
    gabor_feats = []
    for feature in features:
        padded = pad_to_square(feature)
        dim = int(np.sqrt(padded.size))
        image = padded.reshape(dim, dim)
        responses = []
        for freq in frequencies:
            for theta in orientations:
                real, imag = gabor(image, frequency=freq, theta=theta)
                magnitude = np.hypot(real, imag)
                flat = magnitude.flatten()
                if flat.std() > 0:
                    flat = (flat - flat.mean()) / flat.std()
                responses.append(flat)
        gabor_feats.append(np.concatenate(responses))
    return np.asarray(gabor_feats)


def compute_scores(train_features_path: str, eval_features_path: str) -> None:
    """Compute LDA accuracy and energy score using raw embeddings."""
    train = torch.load(train_features_path)
    evald = torch.load(eval_features_path)
    X_train = to_numpy(train["embeddings"])
    y_train = to_numpy(train["labels"])
    X_eval = to_numpy(evald["embeddings"])
    y_eval = to_numpy(evald["labels"])

    lda = LDA(shrinkage=0.1)
    lda.fit(X_train, y_train)
    probs = lda.predict_proba(X_eval)
    lda_score = probs[np.arange(len(y_eval)), y_eval].mean()
    logits = X_eval @ lda.coef_.T + lda.intercept_
    energy = Energy_Score(logits, percent=100, tail="bot")
    print(f"LDA Score: {lda_score}")
    print(f"Energy Score (Full): {energy}")


def compute_gabor_scores(train_features_path: str, eval_features_path: str) -> None:
    """Compute energy score using Gabor features for local texture."""
    train = torch.load(train_features_path)
    evald = torch.load(eval_features_path)
    X_train = to_numpy(train["embeddings"])
    y_train = to_numpy(train["labels"])
    X_eval = to_numpy(evald["embeddings"])
    y_eval = to_numpy(evald["labels"])

    X_train_gabor = compute_gabor_features(X_train)
    X_eval_gabor = compute_gabor_features(X_eval)

    lda = LDA(shrinkage=0.1)
    lda.fit(X_train_gabor, y_train)
    logits = X_eval_gabor @ lda.coef_.T + lda.intercept_
    energy = Energy_Score(logits, percent=100, tail="bot")
    print(f"Gabor Energy Score (Full): {energy}")

