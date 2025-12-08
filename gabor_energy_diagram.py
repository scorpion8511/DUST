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
from typing import Dict, Mapping

import matplotlib
import matplotlib.pyplot as plt
import torch

from gabor_eng import compute_gabor_scores

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


def _compute_energy_from_paths(train_path: str, eval_path: str, device: str) -> float:
    train = torch.load(train_path, map_location=device)
    evald = torch.load(eval_path, map_location=device)
    scores = compute_gabor_scores(
        train["embeddings"],
        train["labels"],
        evald["embeddings"],
        evald["labels"],
        device=device,
    )
    return float(scores["energy"])


def compute_energy_scores(
    dataset_model_paths: Dict[str, Dict[str, Mapping[str, str]]],
    device: str,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
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
            energy = _compute_energy_from_paths(train_path, eval_path, device)
            results[dataset][model_name] = energy
            print(f"  Gabor Energy:   {energy:.6f}")
    return results


def plot_energy_diagram(scores: Dict[str, Dict[str, float]], output: pathlib.Path) -> None:
    datasets = list(scores.keys())
    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 5), squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        models = list(scores[dataset].keys())
        energies = [scores[dataset][m] for m in models]
        bars = ax.bar(models, energies, color="#4C72B0")
        ax.set_title(f"{dataset} Gabor Energy")
        ax.set_ylabel("Energy score")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=45, labelrotation=45)
        for bar, energy in zip(bars, energies):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{energy:.3f}",
                ha="center",
                va="bottom",
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
    args = parser.parse_args()

    selected = {k: v for k, v in DEFAULT_DATASET_MODEL_PATHS.items() if k in args.dataset}
    if not selected:
        raise ValueError("No datasets selected. Check the --dataset argument.")

    scores = compute_energy_scores(selected, device=args.device)
    plot_energy_diagram(scores, args.output)


if __name__ == "__main__":
    main()
