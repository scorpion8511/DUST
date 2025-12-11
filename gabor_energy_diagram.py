"""Plot Gabor energy scores across datasets/models for unimodal features.

This helper mirrors the Gabor energy computation used in ``scoring_pipeline.py``
while focusing purely on the energy metric for unimodal models. It loads the
train/eval feature pairs for each dataset and model, computes the training-free
Gabor energy, and writes a grouped bar chart to disk (default:
``gabor_energy.png``). Example usage:

    python gabor_energy_diagram.py --output gabor_energy.png --device cpu
    python gabor_energy_diagram.py --dataset TCGA CAM

The defaults include the TCGA unimodal feature paths from previous scripts so
the diagram can be generated without extra configuration.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List, Mapping

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from gabor_eng import (
    LDA,
    compute_gabor_features,
    to_tensor,
    _apply_pca,
)

# Use a non-interactive backend so plots can be generated headlessly.
matplotlib.use("Agg")


DEFAULT_DATASET_MODEL_PATHS: Dict[str, Dict[str, Mapping[str, str]]] = {
    "TCGA": {
        "uni": {
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/uni_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/uni_eval_features.pth",
        },
        "conch": {
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/conch_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/conch_eval_features.pth",
        },
        "giga": {
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/giga_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/giga_eval_features.pth",
        },
        "phikon": {
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/phikon_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/phikon_eval_features.pth",
        },
        "virchow": {
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_eval_features.pth",
        },
    }
}

DEFAULT_ACCURACIES: Dict[str, Dict[str, float]] = {
    "TCGA": {
        "uni": 0.6856,
        "conch": 0.6916,
        "giga": 0.7108,
        "phikon": 0.6675,
        "virchow": 0.6952,
    }
}


def _compute_energy_distribution_from_paths(
    train_path: str,
    eval_path: str,
    device: str,
    pca_dim: int = 128,
    sample_fraction: float = 0.5,
) -> List[float]:
    """Return per-sample Gabor energy scores for a train/eval feature pair."""

    train = torch.load(train_path, map_location=device)
    evald = torch.load(eval_path, map_location=device)

    def _subsample(tensor: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        count = max(1, int(round(tensor.shape[0] * sample_fraction)))
        idx = torch.randperm(tensor.shape[0], device=tensor.device)[:count]
        return tensor[idx], labels[idx]

    train_embeddings = to_tensor(train["embeddings"], device)
    eval_embeddings = to_tensor(evald["embeddings"], device)
    train_labels = to_tensor(train["labels"], device).long()
    eval_labels = to_tensor(evald["labels"], device).long()

    train_embeddings, train_labels = _subsample(train_embeddings, train_labels)
    eval_embeddings, eval_labels = _subsample(eval_embeddings, eval_labels)

    train_feats = compute_gabor_features(train_embeddings, device=device)
    eval_feats = compute_gabor_features(eval_embeddings, device=device)

    mean = train_feats.mean(0, keepdim=True)
    std = train_feats.std(0, keepdim=True).clamp_min(1e-6)
    train_feats = (train_feats - mean) / std
    eval_feats = (eval_feats - mean) / std

    if pca_dim:
        pca_dim = min(pca_dim, train_feats.shape[0], train_feats.shape[1])
        if pca_dim > 0:
            train_feats, eval_feats = _apply_pca(train_feats, eval_feats, pca_dim)

    lda = LDA(shrinkage=0.1, device=device)
    lda.fit(train_feats, train_labels)
    logits = eval_feats @ lda.coef_.T + lda.intercept_

    energies = torch.logsumexp(logits, dim=-1)
    return energies.cpu().tolist()


def compute_energy_distributions(
    dataset_model_paths: Dict[str, Dict[str, Mapping[str, str]]],
    device: str,
    sample_fraction: float,
) -> Dict[str, Dict[str, List[float]]]:
    results: Dict[str, Dict[str, List[float]]] = {}
    for dataset, models in dataset_model_paths.items():
        results[dataset] = {}
        print(f"\n=== Dataset: {dataset} ===")
        for model_name, paths in models.items():
            train_path = paths.get("train") or paths.get("eval")
            eval_path = paths.get("eval") or paths.get("train")
            if not train_path or not eval_path:
                raise ValueError(f"Model '{model_name}' in dataset '{dataset}' is missing feature paths")
            print(f"Processing model: {model_name}")
            print(f"  Train features: {train_path}")
            print(f"  Eval features:  {eval_path}")
            print(f"  Sampling fraction: {sample_fraction:.2f} of available embeddings")
            energies = _compute_energy_distribution_from_paths(
                train_path, eval_path, device, sample_fraction=sample_fraction
            )
            results[dataset][model_name] = energies
            print(
                f"  Gabor Energy:   mean={float(torch.tensor(energies).mean()):.4f},",
                f" std={float(torch.tensor(energies).std()):.4f}"
            )
    return results

def _clip_distributions_to_mean(
    distributions: Dict[str, Dict[str, List[float]]]
) -> tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, float]]]:
    """Clamp per-model energies to their mean and record the original means."""

    clipped: Dict[str, Dict[str, List[float]]] = {}
    means: Dict[str, Dict[str, float]] = {}
    for dataset, models in distributions.items():
        clipped[dataset] = {}
        means[dataset] = {}
        for model, values in models.items():
            t_values = torch.tensor(values)
            mean_val = float(t_values.mean()) if t_values.numel() else 0.0
            means[dataset][model] = mean_val
            clipped_vals = torch.clamp(t_values, max=mean_val).tolist()
            clipped[dataset][model] = clipped_vals

    return clipped, means


def _compute_global_bins(
    distributions: Dict[str, Dict[str, List[float]]], bin_count: int = 40
) -> torch.Tensor:
    """Return shared histogram bin edges across every dataset/model."""

    all_values: List[float] = []
    for models in distributions.values():
        for values in models.values():
            all_values.extend(values)

    if not all_values:
        return torch.linspace(0.0, 1.0, bin_count + 1)

    min_v = min(all_values)
    max_v = max(all_values)
    if max_v - min_v < 1e-6:
        max_v = min_v + 1.0

    # Pad the range slightly so means and tails are visible together.
    pad = 0.02 * (max_v - min_v)
    return torch.linspace(min_v - pad, max_v + pad, bin_count + 1)


def plot_energy_diagram(distributions: Dict[str, Dict[str, List[float]]], output: pathlib.Path) -> None:
    """Draw per-model Gabor energy histograms and a companion mean-energy bar plot."""

    datasets = list(distributions.keys())
    num_datasets = len(datasets)
    fig, axes = plt.subplots(
        2,
        num_datasets,
        figsize=(8 * num_datasets, 7.5),
        squeeze=False,
        gridspec_kw={"height_ratios": [3.2, 1.4]},
    )

    clipped_distributions, mean_lookup = _clip_distributions_to_mean(distributions)
    bins = _compute_global_bins(clipped_distributions)
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#64B5CD", "#CCB974"]

    for col_idx, dataset in enumerate(datasets):
        hist_ax = axes[0][col_idx]
        bar_ax = axes[1][col_idx]
        models = list(distributions[dataset].keys())
        if not models:
            continue

        handles = []
        labels = []
        mean_values: List[float] = []
        colors: List[str] = []

        for j, model in enumerate(models):
            values = torch.tensor(clipped_distributions[dataset][model])
            original_mean = mean_lookup[dataset][model]
            color = palette[j % len(palette)]
            _, _, patches = hist_ax.hist(
                values.numpy(),
                bins=bins.numpy(),
                density=True,
                alpha=0.4,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=None,
            )

            mean_val = original_mean
            handle = patches[0] if patches else None
            if handle:
                handles.append(handle)
                if dataset in DEFAULT_ACCURACIES and model in DEFAULT_ACCURACIES[dataset]:
                    labels.append(f"{model} (acc={DEFAULT_ACCURACIES[dataset][model]:.4f})")
                else:
                    labels.append(model)

            mean_values.append(mean_val)
            colors.append(color)

        hist_ax.set_title(dataset)
        hist_ax.set_xlabel("Gabor Energy")
        if col_idx == 0:
            hist_ax.set_ylabel("Density")
        hist_ax.legend(handles, labels, frameon=False)
        hist_ax.grid(alpha=0.2, linestyle=":", linewidth=0.5)

        # Companion mean-energy bars for quick model comparison.
        bar_positions = range(len(models))
        bars = bar_ax.bar(
            bar_positions, mean_values, color=colors, alpha=0.8, width=0.55
        )
        bar_ax.set_xticks(list(bar_positions))
        bar_ax.set_xticklabels(models, rotation=20, ha="right")
        bar_ax.set_ylabel("Mean energy")
        bar_ax.grid(alpha=0.2, linestyle=":", linewidth=0.5, axis="y")

        bar_handles: List[mpatches.Patch] = []
        for model, bar, mean_val in zip(models, bars, mean_values):
            bar_handles.append(
                mpatches.Patch(
                    color=bar.get_facecolor(), label=f"{model} (mean={mean_val:.2f})"
                )
            )
        bar_ax.legend(
            handles=bar_handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.3),
            ncol=2,
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    print(f"Saved Gabor energy diagram to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Gabor energy scores for unimodal models")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=list(DEFAULT_DATASET_MODEL_PATHS.keys()),
        help="Datasets to include (default: all configured)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("gabor_energy.png"),
        help="Path to save the bar chart PNG",
    )
    parser.add_argument("--device", default="cpu", help="Device to run scoring on (cpu or cuda)")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.5,
        help="Fraction of embeddings to sample at random (per split) when computing energy",
    )
    args = parser.parse_args()

    selected = {k: v for k, v in DEFAULT_DATASET_MODEL_PATHS.items() if k in args.dataset}
    if not selected:
        raise ValueError("No datasets selected. Check the --dataset argument.")

    scores = compute_energy_distributions(
        selected, device=args.device, sample_fraction=args.sample_fraction
    )
    plot_energy_diagram(scores, args.output)


if __name__ == "__main__":
    main()
