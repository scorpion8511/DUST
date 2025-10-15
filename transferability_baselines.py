"""Utilities for evaluating feature transferability baselines.

This module centralises the ground-truth accuracy tables and feature
registries that the CLI uses to score transferability metrics.  The
defaults are intentionally lightweight – they primarily act as documented
examples that can be customised via the command line.

The module also provides helpers for parsing feature overrides and for
computing Kendall's tau between metric-derived rankings and reference
accuracies.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from scipy.stats import weightedtau as _scipy_weightedtau
except ImportError:  # pragma: no cover - exercised via fallback path in tests
    _scipy_weightedtau = None


# ---------------------------------------------------------------------------
# Ground-truth accuracy tables
# ---------------------------------------------------------------------------

#: Default ground-truth accuracies for common datasets.
#:
#: The TCGA numbers match the figures distributed internally with the original
#: request.  Additional datasets include deliberately small toy entries so that
#: tests and quick smoke runs have something to work with by default.  Users
#: can provide their own tables by importing this module and updating the
#: dictionary in place before invoking the CLI.
GROUND_TRUTH_ACCURACIES: Dict[str, Dict[str, float]] = {
    "tcga": {
        "uni": 0.936,
        "conch": 0.949,
        "giga": 0.921,
        "phikon": 0.914,
        "virchow2": 0.905,
    },
    "lc": {
        "uni": 0.940,
        "conch": 0.932,
        "giga": 0.928,
        "phikon": 0.917,
        "virchow2": 0.909,
    },
}


# ---------------------------------------------------------------------------
# Dataset → feature registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeaturePaths:
    """Container describing where to load features for a given model."""

    train: Path
    eval: Path

    @classmethod
    def from_string(cls, spec: str) -> "FeaturePaths":
        """Create an instance from ``train[,eval]`` style strings."""

        train_str, *maybe_eval = spec.split(",")
        if maybe_eval:
            eval_str = maybe_eval[0]
        else:
            eval_str = train_str
        return cls(train=Path(train_str).expanduser(), eval=Path(eval_str).expanduser())


def _registry_entry(train: str, eval: Optional[str] = None) -> FeaturePaths:
    return FeaturePaths(Path(train), Path(eval or train))


#: Default registry mapping datasets to models and their feature file paths.
DATASET_FEATURE_REGISTRY: Dict[str, Dict[str, FeaturePaths]] = {
    "tcga": {
        "uni": _registry_entry("features/tcga/uni_train_features.pth", "features/tcga/uni_eval_features.pth"),
        "conch": _registry_entry(
            "features/tcga/conch_train_features.pth", "features/tcga/conch_eval_features.pth"
        ),
        "giga": _registry_entry("features/tcga/giga_train_features.pth", "features/tcga/giga_eval_features.pth"),
        "phikon": _registry_entry(
            "features/tcga/phikon_train_features.pth", "features/tcga/phikon_eval_features.pth"
        ),
        "virchow2": _registry_entry(
            "features/tcga/virchow2_train_features.pth", "features/tcga/virchow2_eval_features.pth"
        ),
    },
    "lc": {
        "uni": _registry_entry("features/lc/uni_train_features.pth", "features/lc/uni_eval_features.pth"),
        "conch": _registry_entry("features/lc/conch_train_features.pth", "features/lc/conch_eval_features.pth"),
        "giga": _registry_entry("features/lc/giga_train_features.pth", "features/lc/giga_eval_features.pth"),
        "phikon": _registry_entry("features/lc/phikon_train_features.pth", "features/lc/phikon_eval_features.pth"),
        "virchow2": _registry_entry("features/lc/virchow2_train_features.pth", "features/lc/virchow2_eval_features.pth"),
    },
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def dataset_ground_truth(dataset: str) -> Dict[str, float]:
    """Return the ground-truth mapping for *dataset* (empty if unknown)."""

    return dict(GROUND_TRUTH_ACCURACIES.get(dataset, {}))


def collect_ground_truth(datasets: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Filter :data:`GROUND_TRUTH_ACCURACIES` for the selected datasets."""

    truth: Dict[str, Dict[str, float]] = {}
    for dataset in datasets:
        gt = dataset_ground_truth(dataset)
        if gt:
            truth[dataset] = gt
    return truth


def build_dataset_model_paths(
    datasets: Iterable[str],
    overrides: Mapping[str, Dict[str, FeaturePaths]] | None = None,
) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """Create a nested ``dataset -> model -> (train, eval)`` mapping."""

    overrides = overrides or {}
    resolved: Dict[str, Dict[str, Tuple[str, str]]] = {}

    for dataset in datasets:
        models = DATASET_FEATURE_REGISTRY.get(dataset, {})
        if not models and dataset not in overrides:
            continue
        resolved[dataset] = {}
        for model, paths in models.items():
            resolved[dataset][model] = (str(paths.train), str(paths.eval))

    for dataset, models in overrides.items():
        if dataset not in resolved:
            resolved[dataset] = {}
        for model, paths in models.items():
            resolved[dataset][model] = (str(paths.train), str(paths.eval))

    return resolved


def parse_feature_override(spec: str) -> Tuple[str, str, FeaturePaths]:
    """Parse a ``DATASET:MODEL=train[,eval]`` override specification."""

    if "=" not in spec:
        raise ValueError("Override must contain '=' separating model and paths")
    lhs, rhs = spec.split("=", 1)
    if ":" not in lhs:
        raise ValueError("Override must specify dataset and model as DATASET:MODEL")
    dataset, model = lhs.split(":", 1)
    paths = FeaturePaths.from_string(rhs)
    return dataset.strip(), model.strip(), paths


def overrides_from_specs(specs: Sequence[str]) -> Dict[str, Dict[str, FeaturePaths]]:
    """Convert CLI override strings into the nested mapping structure."""

    overrides: Dict[str, Dict[str, FeaturePaths]] = {}
    for spec in specs:
        dataset, model, paths = parse_feature_override(spec)
        overrides.setdefault(dataset, {})[model] = paths
    return overrides


def kendall_tau_table(
    metric_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    ground_truth: Mapping[str, Mapping[str, float]],
    metric_signs: Mapping[str, float],
) -> Dict[str, Dict[str, float]]:
    """Compute Kendall's tau for each dataset and metric.

    Parameters
    ----------
    metric_scores:
        Nested mapping ``dataset -> model -> metrics``.  The innermost mapping
        must contain the keys referenced in ``metric_signs``.
    ground_truth:
        Ground-truth accuracy tables matching the datasets in
        ``metric_scores``.  Datasets without ground-truth information are
        ignored.
    metric_signs:
        Orientation factors per metric so that higher values always imply
        better transferability.

    Returns
    -------
    dict
        ``{dataset: {metric: tau}}``.

    Examples
    --------
    >>> metric_scores = {
    ...     "toy": {
    ...         "a": {"combined": 0.8},
    ...         "b": {"combined": 0.5},
    ...     }
    ... }
    >>> truth = {"toy": {"a": 0.9, "b": 0.7}}
    >>> kendall_tau_table(metric_scores, truth, {"combined": 1.0})
    {'toy': {'combined': 1.0}}
    """

    taus: Dict[str, Dict[str, float]] = {}
    for dataset, models in metric_scores.items():
        gt = ground_truth.get(dataset)
        if not gt:
            continue
        taus[dataset] = {}
        for metric, sign in metric_signs.items():
            preds = []
            truth = []
            for model, scores in models.items():
                if model not in gt:
                    continue
                if metric not in scores:
                    continue
                preds.append(sign * scores[metric])
                truth.append(gt[model])
            if len(preds) >= 2:
                tau = weighted_tau(preds, truth)
            elif preds:
                tau = 1.0
            else:
                tau = float("nan")
            taus[dataset][metric] = float(tau)
    return taus


def mean_tau(taus: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Compute per-metric averages across datasets."""

    acc: MutableMapping[str, list[float]] = {}
    for dataset_values in taus.values():
        for metric, value in dataset_values.items():
            if value == value:  # filter NaN
                acc.setdefault(metric, []).append(value)

    return {metric: float(sum(values) / len(values)) for metric, values in acc.items() if values}


def serialise_results(
    dataset_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    taus: Mapping[str, Mapping[str, float]],
    outfile: Path,
) -> None:
    """Serialise scores and taus as JSON to ``outfile``."""

    payload = {
        "scores": dataset_scores,
        "kendall_tau": taus,
        "kendall_tau_mean": mean_tau(taus),
    }
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(payload, indent=2, sort_keys=True))


__all__ = [
    "GROUND_TRUTH_ACCURACIES",
    "DATASET_FEATURE_REGISTRY",
    "FeaturePaths",
    "dataset_ground_truth",
    "collect_ground_truth",
    "build_dataset_model_paths",
    "parse_feature_override",
    "overrides_from_specs",
    "weighted_tau",
    "kendall_tau_table",
    "mean_tau",
    "serialise_results",
]

def _kendall_tau_fallback(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute a simple Kendall tau when SciPy is unavailable."""

    n = len(xs)
    if n != len(ys):
        raise ValueError("Inputs must have the same length")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            x_diff = xs[i] - xs[j]
            y_diff = ys[i] - ys[j]
            if x_diff == 0 or y_diff == 0:
                continue
            concordant += int(x_diff * y_diff > 0)
            discordant += int(x_diff * y_diff < 0)
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def weighted_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return the (weighted) Kendall tau correlation coefficient."""

    if _scipy_weightedtau is None:
        return _kendall_tau_fallback(xs, ys)
    tau, _ = _scipy_weightedtau(xs, ys)
    return float(tau)
