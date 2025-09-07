"""Pipeline for evaluating models with MSCI-V scores.

This script loads multi-magnification feature embeddings from disk, computes
MSCI-V for each model/dataset pair and performs normalisation and global weight
search across all datasets.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from msci import msci_v


def load_multi_magnification_features(root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load features organised as ``root/dataset/model/*.npy``.

    Each ``.npy`` file inside a model directory represents embeddings from a
    particular magnification with shape ``(regions, dim)``.  The files are
    stacked along a new dimension to form ``(regions, mags, dim)``.
    """
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        data[dataset_dir.name] = {}
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            mags = [np.load(f) for f in sorted(model_dir.glob("*.npy"))]
            if not mags:
                continue
            stacked = np.stack(mags, axis=1)  # (regions, mags, dim)
            data[dataset_dir.name][model_dir.name] = stacked
    return data


def compute_msci_scores(features: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for dataset, models in features.items():
        scores[dataset] = {}
        for model, feats in models.items():
            scores[dataset][model] = msci_v(feats)
    return scores


def zscore_across_datasets(scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    all_scores = [score for ds in scores.values() for score in ds.values()]
    mean = float(np.mean(all_scores))
    std = float(np.std(all_scores)) or 1.0
    normalised: Dict[str, Dict[str, float]] = {}
    for dataset, models in scores.items():
        normalised[dataset] = {}
        for model, score in models.items():
            normalised[dataset][model] = (score - mean) / std
    return normalised


def derive_optimal_weights_global(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    targets: Dict[str, Dict[str, float]],
    grid: Iterable[float] = np.linspace(0, 1, 11),
) -> Tuple[Dict[str, float], float]:
    """Perform a global grid search for metric weights.

    Parameters
    ----------
    metrics:
        Nested dictionary ``dataset -> model -> metric_name -> score``.
    targets:
        Nested dictionary with the same dataset/model keys mapping to target
        values (e.g. accuracy).
    grid:
        Iterable of weight values to search.  The search enforces that weights
        sum to one.
    """
    metric_names = sorted(next(iter(next(iter(metrics.values())).values())).keys())
    X: List[List[float]] = []
    y: List[float] = []
    for dataset, models in metrics.items():
        for model, mvals in models.items():
            X.append([mvals[m] for m in metric_names])
            y.append(targets[dataset][model])
    X_arr = np.array(X)
    y_arr = np.array(y)

    best_corr = -np.inf
    best_w: Tuple[float, ...] | None = None
    for weights in itertools.product(grid, repeat=len(metric_names)):
        if not np.isclose(sum(weights), 1.0):
            continue
        pred = X_arr.dot(np.array(weights))
        corr = np.corrcoef(pred, y_arr)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_w = weights
    if best_w is None:
        raise RuntimeError("no valid weight combination found")
    return dict(zip(metric_names, best_w)), float(best_corr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSCI-V scoring pipeline")
    parser.add_argument("features_root", type=Path, help="Root directory with features")
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="Optional CSV with columns dataset,model,score for weight search",
    )
    args = parser.parse_args()

    features = load_multi_magnification_features(args.features_root)
    raw_scores = compute_msci_scores(features)
    norm_scores = zscore_across_datasets(raw_scores)

    print("Raw MSCI-V scores:")
    for ds, models in raw_scores.items():
        for model, score in models.items():
            print(f"{ds}/{model}: {score:.4f}")

    print("\nZ-scored across datasets:")
    for ds, models in norm_scores.items():
        for model, score in models.items():
            print(f"{ds}/{model}: {score:.4f}")

    if args.targets and args.targets.exists():
        import csv

        targets: Dict[str, Dict[str, float]] = {}
        with args.targets.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                targets.setdefault(row["dataset"], {})[row["model"]] = float(row["score"])
        metrics = {ds: {m: {"msci_v": s} for m, s in models.items()} for ds, models in norm_scores.items()}
        weights, corr = derive_optimal_weights_global(metrics, targets)
        print("\nOptimal weights:", weights)
        print("Correlation:", corr)


if __name__ == "__main__":
    main()
