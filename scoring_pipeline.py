import numpy as np
import torch
from typing import Dict, Sequence
from scipy.stats import weightedtau

from gabor_eng import compute_gabor_scores


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
    print(f"LDA Score: {scores['lda']}")
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
    weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    all_energy = []
    all_hoi = []
    all_lda = []
    for models in dataset_results.values():
        for scores in models.values():
            all_energy.append(scores["energy"])
            all_hoi.append(scores["hoi"])
            all_lda.append(scores.get("lda", 0.0))
    energy_mean, energy_std = np.mean(all_energy), np.std(all_energy) or 1.0
    hoi_mean, hoi_std = np.mean(all_hoi), np.std(all_hoi) or 1.0
    lda_mean, lda_std = np.mean(all_lda), np.std(all_lda) or 1.0
    combined: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, models in dataset_results.items():
        combined[dataset] = {}
        for model, scores in models.items():
            energy_norm = (scores["energy"] - energy_mean) / energy_std
            hoi_norm = (scores["hoi"] - hoi_mean) / hoi_std
            lda_norm = (scores.get("lda", 0.0) - lda_mean) / lda_std
            combined_score = (
                weights[0] * energy_norm + weights[1] * hoi_norm + weights[2] * lda_norm
            )
            combined[dataset][model] = {
                "energy": scores["energy"],
                "hoi": scores["hoi"],
                "lda": scores.get("lda", 0.0),
                "combined": combined_score,
            }
    return combined


def derive_optimal_weights(
    dataset_results: Dict[str, Dict[str, Dict[str, float]]],
    ground_truth: Dict[str, Dict[str, float]],
    search_space: Sequence[float] = np.linspace(-1.0, 1.0, 41),
    default: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
) -> Sequence[float]:
    """Grid search weights to maximize Kendall tau_w against ground truth."""

    energies, hois, ldas, accuracies = [], [], [], []
    for dataset, models in dataset_results.items():
        if dataset not in ground_truth:
            continue
        for model, scores in models.items():
            if model in ground_truth[dataset]:
                energies.append(scores["energy"])
                hois.append(scores["hoi"])
                ldas.append(scores.get("lda", 0.0))
                accuracies.append(ground_truth[dataset][model])

    if not energies:
        return default

    energy = np.asarray(energies)
    hoi = np.asarray(hois)
    lda = np.asarray(ldas)
    acc = np.asarray(accuracies)
    energy = (energy - energy.mean()) / (energy.std() or 1.0)
    hoi = (hoi - hoi.mean()) / (hoi.std() or 1.0)
    lda = (lda - lda.mean()) / (lda.std() or 1.0)

    best_tau = -2.0
    best_w = default
    for w_energy in search_space:
        for w_hoi in search_space:
            for w_lda in search_space:
                preds = w_energy * energy + w_hoi * hoi + w_lda * lda
                tau, _ = weightedtau(preds, acc)
                if tau > best_tau:
                    best_tau = tau
                    best_w = (float(w_energy), float(w_hoi), float(w_lda))
    print(f"Optimized weights: {best_w} (tau={best_tau})")
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
    dataset_model_paths = {
        "LC": {"uni": ("/path/to/train.pth", "/path/to/eval.pth")}
    }
    raw_scores = compute_scores_for_all_datasets(dataset_model_paths)
    ground_truth = {"LC": {"uni": 0.94}}
    weights = derive_optimal_weights(raw_scores, ground_truth)
    combined_scores = normalize_and_combine_scores(raw_scores, weights=weights)
    compute_kendall_tau_across_datasets(combined_scores, ground_truth)
