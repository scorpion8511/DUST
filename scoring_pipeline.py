import argparse
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import weightedtau

from gabor_eng import compute_gabor_scores, benchmark_runtime


@dataclass
class MSCIResult:
    score: float
    mean_variance: float
    max_variance: float
    per_region_variance: Sequence[float]
    regions_used: int
    regions_total: int


def _normalise_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def _canonicalise_magnification(value: Any) -> str:
    text = str(value).strip().replace("×", "x")
    if text.lower().endswith("x"):
        text = text[:-1]
    return text


def _prepare_magnification_key(value: Any) -> tuple[float, str]:
    text = _canonicalise_magnification(value)
    try:
        numeric = float(text)
    except ValueError:
        return (float("inf"), _canonicalise_magnification(value))
    return (numeric, _canonicalise_magnification(value))


def compute_msci_single_modality(
    eval_features: Mapping[str, Any],
    manifest_source: str | pd.DataFrame,
    *,
    region_column: str = "region_id",
    magnification_column: str = "magnification",
    magnifications: Sequence[Any] | None = None,
) -> MSCIResult:
    """Compute single-modality MSCI using evaluation embeddings and a manifest.

    The manifest must align with the order of embeddings in ``eval_features`` and
    contain one row per patch that includes the region identifier and
    magnification. For each region, embeddings are averaged per magnification,
    cosine-normalised, and compared against the region centroid to measure
    cross-scale consistency.
    """

    if "embeddings" not in eval_features:
        raise KeyError("Evaluation features must contain an 'embeddings' entry")

    embeddings_tensor = eval_features["embeddings"]
    if isinstance(embeddings_tensor, torch.Tensor):
        embeddings = embeddings_tensor.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings_tensor)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array of shape (N, D)")

    if isinstance(manifest_source, pd.DataFrame):
        manifest = manifest_source.reset_index(drop=True)
    else:
        manifest = pd.read_csv(manifest_source)

    # If the manifest contains more rows than the embeddings (for example when
    # aggregating across splits), align rows using the optional ``row_index``
    # column that preserves the manifest position for each embedding.
    if len(manifest) != embeddings.shape[0]:
        if "row_index" in manifest and "row_indices" in eval_features:
            row_index_series = pd.Series(eval_features["row_indices"], name="row_index")
            manifest = (
                manifest.merge(row_index_series.to_frame(), on="row_index", how="right")
                .reset_index(drop=True)
            )
        elif "__row_index" in manifest and "row_indices" in eval_features:
            row_index_series = pd.Series(eval_features["row_indices"], name="__row_index")
            manifest = (
                manifest.merge(row_index_series.to_frame(), on="__row_index", how="right")
                .reset_index(drop=True)
            )

    if len(manifest) != embeddings.shape[0]:
        raise ValueError(
            "Manifest row count does not match number of embeddings: "
            f"{len(manifest)} vs {embeddings.shape[0]}"
        )

    if magnifications is not None:
        required = [_canonicalise_magnification(m) for m in magnifications]
    else:
        required = None

    region_groups = manifest.groupby(region_column, sort=False)
    per_region_vars: list[float] = []

    for region, group in region_groups:
        idx = group.index.to_numpy()
        mags = group[magnification_column].apply(_canonicalise_magnification).tolist()
        mag_to_embeddings: Dict[str, list[np.ndarray]] = {}
        for row_idx, mag in zip(idx, mags):
            mag_to_embeddings.setdefault(mag, []).append(embeddings[row_idx])

        if required is not None and not set(required).issubset(mag_to_embeddings):
            continue

        ordered_mags = (
            required
            if required is not None
            else [mag for _, mag in sorted((_prepare_magnification_key(m), m) for m in mag_to_embeddings)]
        )

        mag_vectors: list[np.ndarray] = []
        for mag in ordered_mags:
            vectors = mag_to_embeddings.get(mag)
            if not vectors:
                continue
            mean_vec = np.mean(np.stack(vectors, axis=0), axis=0)
            mag_vectors.append(_normalise_vector(mean_vec))

        if len(mag_vectors) < 2:
            continue

        centroid = _normalise_vector(np.mean(mag_vectors, axis=0))
        similarities = np.array([float(np.dot(vec, centroid)) for vec in mag_vectors])
        variance = float(np.var(similarities, ddof=0))
        per_region_vars.append(variance)

    regions_used = len(per_region_vars)
    regions_total = region_groups.ngroups
    if regions_used == 0:
        raise ValueError(
            "No regions with the required magnifications were found to compute MSCI"
        )

    mean_variance = float(np.mean(per_region_vars))
    max_variance = 1.0  # cosine similarities lie in [-1, 1]
    score = 1.0 - min(max(mean_variance / max_variance, 0.0), 1.0)

    return MSCIResult(
        score=score,
        mean_variance=mean_variance,
        max_variance=max_variance,
        per_region_variance=per_region_vars,
        regions_used=regions_used,
        regions_total=regions_total,
    )


def compute_gabor_scores_from_paths(
    train_features_path: str,
    eval_features_path: str,
    device: str = "cpu",
    msci_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    train = torch.load(train_features_path, map_location=device)
    evald = torch.load(eval_features_path, map_location=device)
    scores: Dict[str, Any] = compute_gabor_scores(
        train["embeddings"],
        train["labels"],
        evald["embeddings"],
        evald["labels"],
        device=device,
    )
    print(f"Gabor Energy Score (Full): {scores['energy']}")
    print(f"Gabor Fisher Score: {scores['fisher']}")

    msci_params: Dict[str, Any] = {}
    manifest_source: str | pd.DataFrame | None = None

    def _resolve_column(
        key: str,
        fallback_payloads: Sequence[Mapping[str, Any]],
        default: str,
    ) -> str:
        if msci_config and key in msci_config and msci_config[key]:
            return str(msci_config[key])
        for payload in fallback_payloads:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        return default

    if msci_config:
        manifest_candidate = msci_config.get("manifest") or msci_config.get("manifest_path")
        if manifest_candidate:
            manifest_source = manifest_candidate
        msci_params["region_column"] = msci_config.get("region_column", "region_id")
        msci_params["magnification_column"] = msci_config.get(
            "magnification_column", "magnification"
        )
        if msci_config.get("magnifications") is not None:
            msci_params["magnifications"] = msci_config.get("magnifications")

    region_column = msci_params.get("region_column") or _resolve_column(
        "region_column", (train, evald), "region_id"
    )
    magnification_column = msci_params.get("magnification_column") or _resolve_column(
        "magnification_column", (train, evald), "magnification"
    )
    msci_params.setdefault("region_column", region_column)
    msci_params.setdefault("magnification_column", magnification_column)

    def _build_msci_payload(
        payloads: Sequence[Mapping[str, Any]]
    ) -> tuple[Mapping[str, Any] | None, pd.DataFrame | None]:
        embedding_chunks: list[torch.Tensor] = []
        manifest_frames: list[pd.DataFrame] = []

        for payload in payloads:
            if "embeddings" not in payload:
                continue
            if "region_ids" not in payload or "magnifications" not in payload:
                continue

            embeddings_tensor = payload["embeddings"]
            if isinstance(embeddings_tensor, torch.Tensor):
                tensor = embeddings_tensor.detach().cpu()
            else:
                tensor = torch.as_tensor(embeddings_tensor)

            regions = list(payload["region_ids"])
            magnifications = list(payload["magnifications"])
            if len(regions) != len(magnifications) or len(regions) != tensor.shape[0]:
                print(
                    "Warning: Skipping MSCI chunk because metadata lengths do not match embeddings."
                )
                continue

            frame_data: Dict[str, Sequence[Any]] = {
                region_column: regions,
                magnification_column: magnifications,
            }
            if "row_indices" in payload:
                row_indices = list(payload["row_indices"])
                if len(row_indices) == tensor.shape[0]:
                    frame_data["row_index"] = row_indices
            manifest_frames.append(pd.DataFrame(frame_data))
            embedding_chunks.append(tensor)

        if not embedding_chunks:
            return None, None

        combined_embeddings = torch.cat(embedding_chunks, dim=0)
        combined_manifest = pd.concat(manifest_frames, ignore_index=True)
        feature_payload: Dict[str, Any] = {"embeddings": combined_embeddings}
        if "row_index" in combined_manifest and not combined_manifest["row_index"].isnull().any():
            feature_payload["row_indices"] = combined_manifest["row_index"].astype(int).tolist()
        return feature_payload, combined_manifest

    # Attempt to auto-populate manifest information from the feature payloads,
    # preferring the combined (train+eval) set so that regions with different
    # magnifications across splits are still represented.
    combined_features, combined_manifest = _build_msci_payload((train, evald))
    if manifest_source is None and combined_manifest is not None:
        manifest_source = combined_manifest
        msci_features: Mapping[str, Any] = combined_features or evald
    else:
        msci_features = evald

    # Fall back to using only the evaluation payload when a combined manifest is
    # unavailable (for example if metadata is missing).
    if manifest_source is None and "region_ids" in evald and "magnifications" in evald:
        regions = list(evald["region_ids"])
        magnifications_eval = list(evald["magnifications"])
        if len(regions) == len(magnifications_eval) == len(evald["embeddings"]):
            manifest_df = pd.DataFrame(
                {
                    region_column: regions,
                    magnification_column: magnifications_eval,
                }
            )
            if "row_indices" in evald:
                manifest_df["row_index"] = list(evald["row_indices"])
                if isinstance(msci_features, dict):
                    msci_features.setdefault("row_indices", manifest_df["row_index"].tolist())
            manifest_source = manifest_df
        else:
            print(
                "Warning: Unable to auto-construct MSCI manifest because region/magnification "
                "metadata lengths do not match embeddings."
            )

    if manifest_source is None and "manifest_path" in evald:
        manifest_source = evald["manifest_path"]
        msci_params.setdefault("region_column", evald.get("region_column", "region_id"))
        msci_params.setdefault(
            "magnification_column", evald.get("magnification_column", "magnification")
        )

    if manifest_source is not None:
        result = compute_msci_single_modality(
            msci_features,
            manifest_source,
            region_column=msci_params.get("region_column", "region_id"),
            magnification_column=msci_params.get("magnification_column", "magnification"),
            magnifications=msci_params.get("magnifications"),
        )
        scores["msci"] = result.score
        scores["msci_mean_variance"] = result.mean_variance
        scores["msci_max_variance"] = result.max_variance
        scores["msci_regions_used"] = result.regions_used
        scores["msci_regions_total"] = result.regions_total
        scores["msci_variances"] = list(result.per_region_variance)
        print(
            "MSCI score: "
            f"{result.score:.6f} (mean variance={result.mean_variance:.6f}, "
            f"regions used={result.regions_used}/{result.regions_total})"
        )
        if result.per_region_variance:
            preview = result.per_region_variance[:10]
            print(
                "Per-region variance (first 10 values): "
                + np.array2string(np.asarray(preview), separator=", ")
            )
    else:
        print(
            "MSCI metadata not available; skipping MSCI computation for this model."
        )
    return scores


def _extract_paths_config(
    paths: Any,
) -> tuple[str, str, Mapping[str, Any] | None]:
    if isinstance(paths, Mapping):
        train_path = paths.get("train") or paths.get("path")
        if train_path is None:
            raise ValueError("Model configuration dictionary must include a 'train' path")
        eval_path = paths.get("eval", train_path)
        msci_cfg = paths.get("msci")
        manifest = paths.get("manifest")
        if manifest:
            msci_cfg = dict(msci_cfg or {})
            msci_cfg.setdefault("manifest", manifest)
        return str(train_path), str(eval_path), msci_cfg

    if isinstance(paths, (tuple, list)):
        if len(paths) == 3:
            train_path, eval_path, manifest_path = paths
            return str(train_path), str(eval_path), {"manifest": manifest_path}
        if len(paths) == 2:
            train_path, eval_path = paths
            return str(train_path), str(eval_path), None
        if len(paths) == 1:
            (train_path,) = paths
            return str(train_path), str(train_path), None
        raise ValueError("Model path tuples must have length 1, 2, or 3")

    return str(paths), str(paths), None


def compute_scores_for_all_models(
    model_paths: Dict[str, Any], device: str = "cpu"
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for model_name, paths in model_paths.items():
        print(f"\nProcessing model: {model_name}")
        train_path, eval_path, msci_cfg = _extract_paths_config(paths)
        print(f"Training features: {train_path}")
        print(f"Evaluation features: {eval_path}")
        scores = compute_gabor_scores_from_paths(
            train_path, eval_path, device=device, msci_config=msci_cfg
        )
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
