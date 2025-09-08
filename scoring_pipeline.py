import argparse
import numpy as np
import torch
from typing import Dict, Sequence
from scipy.stats import weightedtau

from gabor_eng import compute_gabor_scores, benchmark_runtime


def compute_gabor_scores_from_paths(
    train_features_path: str, eval_features_path: str, device: str = "cpu"
) -> Dict[str, float]:
    train = torch.load(train_features_path, map_location=device)
    evald = torch.load(eval_features_path, map_location=device)
    scores = compute_gabor_scores(
        train["embeddings"],
        train["labels"],
        evald["embeddings"],
        evald["labels"],
        device=device,
    )
    print(f"Gabor Energy Score (Full): {scores['energy']}")
    print(f"Gabor HoI Score: {scores['hoi']}")
    return scores


def compute_scores_for_all_models(
    model_paths: Dict[str, Sequence[str]], device: str = "cpu"
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for model_name, paths in model_paths.items():
        print(f"\nProcessing model: {model_name}")
        if isinstance(paths, (tuple, list)) and len(paths) == 2:
            train_path, eval_path = paths
        else:
            train_path = eval_path = paths
        print(f"Training features: {train_path}")
        print(f"Evaluation features: {eval_path}")
        scores = compute_gabor_scores_from_paths(train_path, eval_path, device=device)
        results[model_name] = scores
    return results


def compute_scores_for_all_datasets(
    dataset_model_paths: Dict[str, Dict[str, Sequence[str]]], device: str = "cpu"
) -> Dict[str, Dict[str, Dict[str, float]]]:
    dataset_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, model_paths in dataset_model_paths.items():
        print(f"\n=== Dataset: {dataset} ===")
        dataset_results[dataset] = compute_scores_for_all_models(model_paths, device=device)
    return dataset_results


def normalize_and_combine_scores(
    dataset_results: Dict[str, Dict[str, Dict[str, float]]],
    weights: Sequence[float] = (0.5, 0.5),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Normalize metrics within each dataset and form a weighted sum.

    Energy scores are inverted so that lower raw energy yields a higher
    normalized value. HoI scores already follow the convention that larger is
    better. Per-dataset normalization prevents one dataset's scale from
    dominating the combined metric. ``weights`` gives the global weighting for
    normalized energy and HoI respectively.
    """

    w_energy, w_hoi = weights
    combined: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, models in dataset_results.items():
        energies = np.array([s["energy"] for s in models.values()])
        hois = np.array([s["hoi"] for s in models.values()])
        e_mean, e_std = energies.mean(), energies.std() or 1.0
        h_mean, h_std = hois.mean(), hois.std() or 1.0

        combined[dataset] = {}
        for model, scores in models.items():
            energy_norm = -((scores["energy"] - e_mean) / e_std)
            hoi_norm = (scores["hoi"] - h_mean) / h_std
            combined_score = w_energy * energy_norm + w_hoi * hoi_norm
            combined[dataset][model] = {
                "energy": scores["energy"],
                "hoi": scores["hoi"],
                "combined": combined_score,
            }
    return combined


def derive_optimal_weights(
    dataset_results: Dict[str, Dict[str, Dict[str, float]]],
    ground_truth: Dict[str, Dict[str, float]],
    search_space: Sequence[float] = np.linspace(-1.0, 1.0, 41),
    default: Sequence[float] = (0.5, 0.5),
) -> Sequence[float]:
    """Grid-search a single pair of metric weights across all datasets."""

    energy_all, hoi_all, acc_all = [], [], []
    for dataset, models in dataset_results.items():
        if dataset not in ground_truth:
            continue
        energies = []
        hois = []
        accs = []
        for model, scores in models.items():
            if model in ground_truth[dataset]:
                energies.append(scores["energy"])
                hois.append(scores["hoi"])
                accs.append(ground_truth[dataset][model])
        if not energies:
            continue
        energies = np.asarray(energies)
        hois = np.asarray(hois)
        accs = np.asarray(accs)
        energies = -((energies - energies.mean()) / (energies.std() or 1.0))
        hois = (hois - hois.mean()) / (hois.std() or 1.0)
        energy_all.extend(energies)
        hoi_all.extend(hois)
        acc_all.extend(accs)

    if not energy_all:
        return default
    energy_all = np.asarray(energy_all)
    hoi_all = np.asarray(hoi_all)
    acc_all = np.asarray(acc_all)

    best_tau = -2.0
    best_w = default
    for w_energy in search_space:
        for w_hoi in search_space:
            preds = w_energy * energy_all + w_hoi * hoi_all
            tau, _ = weightedtau(preds, acc_all)
            if tau > best_tau:
                best_tau = tau
                best_w = (float(w_energy), float(w_hoi))
    print(f"Optimized global weights: {best_w} (tau={best_tau})")
    return best_w


def compute_weighted_kendall_tau(
    scores: Dict[str, float], ground_truth: Dict[str, float]
) -> float:
    common = [m for m in scores if m in ground_truth]
    pred = [scores[m] for m in common]
    truth = [ground_truth[m] for m in common]
    tau, _ = weightedtau(pred, truth)
    return tau


def compute_kendall_tau_across_datasets(
    all_scores: Dict[str, Dict[str, Dict[str, float]]],
    ground_truth: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    taus: Dict[str, float] = {}
    for dataset, model_scores in all_scores.items():
        if dataset in ground_truth:
            preds = {m: s["combined"] for m, s in model_scores.items()}
            tau = compute_weighted_kendall_tau(preds, ground_truth[dataset])
            taus[dataset] = tau
            print(f"Kendall tau_w for {dataset}: {tau}")
    return taus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score models using Gabor features")
    parser.add_argument("--device", default="cpu", help="Device to run scoring on")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark runtime on CPU vs GPU and print results",
    )
    args = parser.parse_args()

    dataset_model_paths = {
        "LC": {"uni": ("/path/to/train.pth", "/path/to/eval.pth")}
    }
    raw_scores = compute_scores_for_all_datasets(dataset_model_paths, device=args.device)
    ground_truth = {"LC": {"uni": 0.94}}
    weights = derive_optimal_weights(raw_scores, ground_truth)
    combined_scores = normalize_and_combine_scores(raw_scores, weights=weights)
    compute_kendall_tau_across_datasets(combined_scores, ground_truth)

    if args.benchmark:
        print("Benchmark:", benchmark_runtime())
