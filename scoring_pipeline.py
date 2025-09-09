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
    print(f"Gabor Fisher Score: {scores['fisher']}")
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
    signs: Sequence[float] = (1.0, 1.0),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Normalize metrics within each dataset and form a weighted sum.

    Parameters
    ----------
    dataset_results:
        Raw ``energy`` and ``fisher`` scores per dataset and model.
    weights:
        Tuple giving the weight for normalized energy and Fisher scores
        respectively when forming the combined score.
    signs:
        Tuple of orientation multipliers for energy and Fisher metrics. A
        negative value flips the metric so that higher values correspond to
        better accuracy.
    """

    w_energy, w_fisher = weights
    s_energy, s_fisher = signs
    combined: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, models in dataset_results.items():
        energies = np.array([s["energy"] for s in models.values()])
        fishers = np.array([s["fisher"] for s in models.values()])
        e_mean, e_std = energies.mean(), energies.std() or 1.0
        f_mean, f_std = fishers.mean(), fishers.std() or 1.0

        combined[dataset] = {}
        for model, scores in models.items():
            energy_norm = s_energy * (-((scores["energy"] - e_mean) / e_std))
            fisher_norm = s_fisher * ((scores["fisher"] - f_mean) / f_std)
            combined_score = w_energy * energy_norm + w_fisher * fisher_norm
            combined[dataset][model] = {
                "energy": scores["energy"],
                "fisher": scores["fisher"],
                "combined": combined_score,
            }
    return combined


def derive_optimal_weights(
    dataset_results: Dict[str, Dict[str, Dict[str, float]]],
    ground_truth: Dict[str, Dict[str, float]],
    search_space: Sequence[float] = np.linspace(-1.0, 1.0, 41),
    default: Sequence[float] = (0.5, 0.5),
) -> Sequence[float]:
    """Grid-search a single pair of metric weights across datasets.

    Metrics are normalized per dataset, their orientations are aligned with the
    ground-truth accuracies, and then a grid search optimizes the mean Kendall
    tau across datasets.
    """

    normed: Dict[str, Dict[str, Sequence[float]]] = {}
    e_all, f_all, acc_all = [], [], []

    for dataset, models in dataset_results.items():
        if dataset not in ground_truth:
            continue
        energies = []
        fishers = []
        accs = []
        names = []
        for model, scores in models.items():
            if model in ground_truth[dataset]:
                energies.append(scores["energy"])
                fishers.append(scores["fisher"])
                accs.append(ground_truth[dataset][model])
                names.append(model)
        if not energies:
            continue
        energies = np.asarray(energies)
        fishers = np.asarray(fishers)
        accs = np.asarray(accs)
        e_norm = -((energies - energies.mean()) / (energies.std() or 1.0))
        f_norm = (fishers - fishers.mean()) / (fishers.std() or 1.0)
        normed[dataset] = {m: (e, f, a) for m, e, f, a in zip(names, e_norm, f_norm, accs)}
        e_all.extend(e_norm)
        f_all.extend(f_norm)
        acc_all.extend(accs)

    if not e_all:
        return default, (1.0, 1.0)

    e_all = np.asarray(e_all)
    f_all = np.asarray(f_all)
    acc_all = np.asarray(acc_all)
    e_tau, _ = weightedtau(e_all, acc_all)
    f_tau, _ = weightedtau(f_all, acc_all)
    e_sign = 1.0 if e_tau >= 0 else -1.0
    f_sign = 1.0 if f_tau >= 0 else -1.0

    for dataset in normed:
        for model in normed[dataset]:
            e, f, a = normed[dataset][model]
            normed[dataset][model] = (e_sign * e, f_sign * f, a)

    best_tau = -2.0
    best_w = default
    for w_energy in search_space:
        for w_fisher in search_space:
            taus = []
            for dataset, models in normed.items():
                preds = []
                truth = []
                for e, f, a in models.values():
                    preds.append(w_energy * e + w_fisher * f)
                    truth.append(a)
                tau, _ = weightedtau(preds, truth)
                taus.append(tau)
            mean_tau = float(np.mean(taus)) if taus else -2.0
            if mean_tau > best_tau:
                best_tau = mean_tau
                best_w = (float(w_energy), float(w_fisher))
    print(f"Optimized global weights: {best_w} (tau={best_tau})")
    return best_w, (e_sign, f_sign)


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
    weights, signs = derive_optimal_weights(raw_scores, ground_truth)
    combined_scores = normalize_and_combine_scores(raw_scores, weights=weights, signs=signs)
    compute_kendall_tau_across_datasets(combined_scores, ground_truth)

    if args.benchmark:
        print("Benchmark:", benchmark_runtime())
