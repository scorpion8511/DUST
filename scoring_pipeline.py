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
    all_energy = []
    all_hoi = []
    for models in dataset_results.values():
        for scores in models.values():
            all_energy.append(scores["energy"])
            all_hoi.append(scores["hoi"])
    energy_mean, energy_std = np.mean(all_energy), np.std(all_energy) or 1.0
    hoi_mean, hoi_std = np.mean(all_hoi), np.std(all_hoi) or 1.0
    combined: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, models in dataset_results.items():
        combined[dataset] = {}
        for model, scores in models.items():
            energy_norm = (scores["energy"] - energy_mean) / energy_std
            hoi_norm = (scores["hoi"] - hoi_mean) / hoi_std
            combined_score = weights[0] * energy_norm + weights[1] * hoi_norm
            combined[dataset][model] = {
                "energy": scores["energy"],
                "hoi": scores["hoi"],
                "combined": combined_score,
            }
    return combined


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
    combined_scores = normalize_and_combine_scores(raw_scores)
    ground_truth = {"LC": {"uni": 0.94}}
    compute_kendall_tau_across_datasets(combined_scores, ground_truth)
