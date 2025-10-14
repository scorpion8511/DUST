import argparse
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

    try:
        numeric = float(text)
    except ValueError:
        return text

    if np.isclose(numeric, round(numeric)):
        numeric = int(round(numeric))

    return f"{numeric:g}"


def _prepare_magnification_key(value: Any) -> tuple[float, str]:
    text = _canonicalise_magnification(value)
    try:
        numeric = float(text)
    except ValueError:
        return (float("inf"), _canonicalise_magnification(value))
    return (numeric, _canonicalise_magnification(value))


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _coerce_magnification_int(value: Any) -> Tuple[int | None, str]:
    canon = _canonicalise_magnification(value)
    try:
        numeric = float(canon)
    except ValueError:
        return None, canon
    if np.isclose(numeric, round(numeric)):
        numeric = int(round(numeric))
    return int(numeric), canon


_REGION_SUFFIX_PATTERN = re.compile(r"(?:[_\-\s]?(?:\d+(?:\.\d+)?)(?:x|×)?)+$")


def _strip_region_suffix(region_values: Sequence[str]) -> list[str]:
    """Remove trailing magnification tokens from region identifiers.

    Some manifests encode magnifications directly in the region identifier (for
    example ``patch_001_5x_10x``).  MSCI requires consistent region identifiers
    across magnifications, so this helper trims any trailing sequence of
    ``_<mag>``/``-<mag>``/`` <mag>`` tokens, optionally suffixed by ``x`` or
    ``×``.  When no suffix is detected the original identifier is preserved.
    """

    cleaned: list[str] = []
    for region in region_values:
        base = region.strip()
        candidate = _REGION_SUFFIX_PATTERN.sub("", base)
        if candidate:
            cleaned.append(candidate)
        else:
            cleaned.append(region)
    return cleaned


def _max_variance(num_magnifications: int) -> float:
    if num_magnifications <= 0:
        raise ValueError("num_magnifications must be positive.")

    max_var = 0.0
    n = float(num_magnifications)
    for k in range(num_magnifications + 1):
        mean = (2.0 * k - n) / n
        diff_pos = 1.0 - mean
        diff_neg = -1.0 - mean
        var = (k * diff_pos * diff_pos + (num_magnifications - k) * diff_neg * diff_neg) / n
        if var > max_var:
            max_var = var
    return max_var


def _metadata_from_features(
    embeddings: torch.Tensor,
    features: Mapping[str, Any],
    region_column: str,
    magnification_column: str,
    *,
    label_column: str | None = None,
) -> pd.DataFrame | None:
    """Construct a metadata frame directly from the feature payload.

    Feature archives produced by :mod:`extract_features` include ``region_ids``
    and ``magnifications`` arrays that align with the stored embeddings.  When
    those arrays are present we prefer them over an external manifest so that
    MSCI always reflects the exact rows used during feature extraction.
    """

    region_values = features.get("region_ids")
    magnification_values = features.get("magnifications")

    if region_values is None or magnification_values is None:
        return None

    try:
        regions = [str(v) for v in region_values]
    except Exception:  # pragma: no cover - defensive
        return None

    mags = list(magnification_values)
    if len(regions) != len(mags) or len(regions) != embeddings.shape[0]:
        warnings.warn(
            "Feature metadata lengths do not match embeddings; falling back to manifest",
            RuntimeWarning,
        )
        return None

    data: Dict[str, Sequence[Any]] = {
        region_column: regions,
        magnification_column: mags,
    }

    label_values: Sequence[Any] | None = None
    label_name = label_column

    if label_column and label_column in features:
        label_values = features[label_column]
    elif "labels" in features:
        label_values = features["labels"]
        if not label_name:
            label_name = "labels"

    if label_values is not None:
        label_array = np.asarray(label_values)
        if label_array.shape[0] == embeddings.shape[0]:
            data[label_name or "labels"] = label_array.tolist()

    row_indices = features.get("row_indices")
    if row_indices is not None and len(row_indices) == embeddings.shape[0]:
        data["row_index"] = list(row_indices)

    return pd.DataFrame(data)


def _load_manifest_for_features(
    manifest_source: str | pd.DataFrame,
    embeddings: torch.Tensor,
    features: Mapping[str, Any],
    region_column: str,
    magnification_column: str,
) -> pd.DataFrame | None:
    """Return a manifest aligned to ``embeddings`` using stored row indices.

    When feature extraction stores ``row_indices`` we can use them to slice the
    original manifest so that each embedding row is paired with its
    magnification metadata.  If the manifest already matches the embedding
    length we simply reset its index for safety.  ``None`` is returned when the
    manifest cannot be aligned (for example missing columns or incompatible
    lengths), allowing the caller to fall back to feature-provided metadata.
    """

    if isinstance(manifest_source, pd.DataFrame):
        manifest_df = manifest_source.copy()
    else:
        manifest_df = pd.read_csv(manifest_source)

    for required in (region_column, magnification_column):
        if required not in manifest_df.columns:
            warnings.warn(
                f"Manifest is missing required column '{required}'; ignoring provided manifest",
                RuntimeWarning,
            )
            return None

    raw_indices = features.get("row_indices")
    row_indices: np.ndarray | None = None
    if raw_indices is not None:
        try:
            row_indices = np.asarray(raw_indices, dtype=int)
        except Exception:  # pragma: no cover - defensive
            row_indices = None

    if row_indices is not None:
        if row_indices.ndim != 1 or row_indices.size != embeddings.shape[0]:
            warnings.warn(
                "Feature row_indices do not align with embeddings; falling back to feature metadata",
                RuntimeWarning,
            )
            row_indices = None

    if row_indices is not None:
        if (row_indices < 0).any() or (row_indices >= len(manifest_df)).any():
            raise ValueError("Feature row indices fall outside the manifest range")
        subset = manifest_df.iloc[row_indices].reset_index(drop=True)
        subset["row_index"] = row_indices.tolist()
        return subset

    if len(manifest_df) == embeddings.shape[0]:
        subset = manifest_df.reset_index(drop=True)
        subset["row_index"] = list(range(len(subset)))
        return subset

    return None


def _ensure_column(
    frame: pd.DataFrame,
    preferred: str | None,
    *,
    fallbacks: Sequence[str] = (),
    required: bool = True,
    context: str = "",
) -> str | None:
    """Return an available column name, optionally trying fallbacks."""

    candidates = [preferred] if preferred else []
    for fallback in fallbacks:
        if fallback not in candidates:
            candidates.append(fallback)

    for name in candidates:
        if name and name in frame.columns:
            if preferred and name != preferred:
                warnings.warn(
                    f"Column '{preferred}' unavailable; using '{name}' instead for {context or 'MSCI'}.",
                    RuntimeWarning,
                )
            return name

    if required:
        raise ValueError(
            f"Required column '{preferred}' is missing from the manifest and no fallbacks were present."
        )
    return None


def _resolve_manifest(
    features: Mapping[str, Any],
    embeddings_tensor: torch.Tensor,
    manifest_source: str | pd.DataFrame | None,
    region_column: str,
    magnification_column: str,
    *,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, str, str, str | None]:
    """Resolve a manifest aligned with ``embeddings_tensor`` and infer columns."""

    manifest: pd.DataFrame | None = None

    if manifest_source is not None:
        manifest = _load_manifest_for_features(
            manifest_source, embeddings_tensor, features, region_column, magnification_column
        )

    if manifest is None:
        manifest = _metadata_from_features(
            embeddings_tensor,
            features,
            region_column,
            magnification_column,
            label_column=label_column,
        )

    if manifest is None:
        raise ValueError(
            "MSCI metadata unavailable; provide a manifest or ensure region information is stored in the features."
        )

    manifest = manifest.copy()

    if len(manifest) != embeddings_tensor.shape[0]:
        feature_row_indices = features.get("row_indices")
        if feature_row_indices is not None:
            row_idx = np.asarray(feature_row_indices)
            if row_idx.ndim == 1 and row_idx.size == embeddings_tensor.shape[0]:
                if (row_idx < 0).any() or (row_idx >= len(manifest)).any():
                    raise ValueError("Feature row indices fall outside the manifest range")
                manifest = manifest.iloc[row_idx].reset_index(drop=True)
                manifest["row_index"] = row_idx.tolist()
            else:
                warnings.warn(
                    "Feature row_indices do not align with embeddings; ignoring stored indices for MSCI alignment.",
                    RuntimeWarning,
                )

    if len(manifest) != embeddings_tensor.shape[0]:
        raise ValueError(
            "Manifest row count does not match number of embeddings: "
            f"{len(manifest)} vs {embeddings_tensor.shape[0]}"
        )

    manifest = manifest.reset_index(drop=True)

    region_column = _ensure_column(
        manifest,
        region_column,
        fallbacks=("patch_id", "region", "region_id", "bag_id"),
        context="region grouping",
    )
    magnification_column = _ensure_column(
        manifest,
        magnification_column,
        fallbacks=("magnification", "patch_scale", "mag", "scale"),
        context="magnification grouping",
    )
    resolved_label_column = None
    if label_column:
        resolved_label_column = _ensure_column(
            manifest,
            label_column,
            fallbacks=("label", "labels", "subtype"),
            required=False,
            context="label lookup",
        )
    else:
        resolved_label_column = _ensure_column(
            manifest,
            None,
            fallbacks=("label", "labels", "subtype"),
            required=False,
            context="label lookup",
        )

    return manifest, region_column, magnification_column, resolved_label_column


def compute_msci_single_modality(
    train_features: Mapping[str, Any],
    eval_features: Mapping[str, Any],
    manifest_source: str | pd.DataFrame | None,
    *,
    region_column: str = "region_id",
    magnification_column: str = "magnification",
    label_column: str | None = "label",
    magnifications: Sequence[Any] | None = None,
) -> MSCIResult:
    """Compute single-modality MSCI by comparing magnification means per region."""

    if "embeddings" not in eval_features:
        raise KeyError("Evaluation features must contain an 'embeddings' entry")
    if "embeddings" not in train_features:
        raise KeyError("Training features must contain an 'embeddings' entry")

    eval_embeddings = _to_tensor(eval_features["embeddings"])
    train_embeddings = _to_tensor(train_features["embeddings"])

    if eval_embeddings.ndim != 2 or train_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays of shape (N, D)")

    eval_manifest, region_column, magnification_column, _ = _resolve_manifest(
        eval_features,
        eval_embeddings,
        manifest_source,
        region_column,
        magnification_column,
        label_column=label_column,
    )

    def _resolve_train_manifest() -> pd.DataFrame | None:
        try:
            manifest, _, _, _ = _resolve_manifest(
                train_features,
                train_embeddings,
                manifest_source,
                region_column,
                magnification_column,
                label_column=label_column,
            )
            return manifest
        except ValueError:
            manifest = _metadata_from_features(
                train_embeddings,
                train_features,
                region_column,
                magnification_column,
                label_column=label_column,
            )
            if manifest is not None and len(manifest) == train_embeddings.shape[0]:
                return manifest
            return None

    train_manifest = _resolve_train_manifest()

    region_embeddings: Dict[str, Dict[int, list[torch.Tensor]]] = {}
    available_magnifications: set[int] = set()

    def _accumulate(manifest: pd.DataFrame | None, embeddings: torch.Tensor) -> None:
        if manifest is None or embeddings.numel() == 0:
            return
        if len(manifest) != embeddings.shape[0]:
            raise ValueError(
                "Manifest row count does not match number of embeddings: "
                f"{len(manifest)} vs {embeddings.shape[0]}"
            )
        regions = _strip_region_suffix(manifest[region_column].astype(str).tolist())
        mags_raw = manifest[magnification_column].tolist()
        if len(regions) != len(mags_raw):
            raise ValueError("Manifest region and magnification columns are misaligned")
        for idx, (region, mag_value) in enumerate(zip(regions, mags_raw)):
            mag_int, _ = _coerce_magnification_int(mag_value)
            if mag_int is None:
                continue
            vector = embeddings[idx]
            region_embeddings.setdefault(region, {}).setdefault(mag_int, []).append(vector)
            available_magnifications.add(mag_int)

    _accumulate(train_manifest, train_embeddings)
    _accumulate(eval_manifest, eval_embeddings)

    if not region_embeddings:
        raise ValueError("No region information available to compute MSCI")

    if magnifications is not None:
        requested: list[int] = []
        for mag in magnifications:
            mag_int, _ = _coerce_magnification_int(mag)
            if mag_int is None:
                raise ValueError(f"Invalid magnification value: {mag}")
            requested.append(mag_int)
        if not requested:
            raise ValueError("No valid magnification levels provided")
        requested_set = set(requested)
        missing_global = [m for m in requested if m not in available_magnifications]
        if missing_global:
            warnings.warn(
                "Some requested magnifications are missing from the combined metadata: "
                f"{missing_global}. Regions lacking these magnifications will be skipped.",
                RuntimeWarning,
            )
    else:
        requested = sorted(available_magnifications)
        requested_set = set(requested)

    if len(requested_set) < 2:
        raise ValueError("MSCI requires at least two magnification levels")

    coverage_counts: Dict[int, int] = {mag: 0 for mag in requested_set}
    per_region_variances: list[float] = []
    per_region_max_variances: list[float] = []
    per_region_norms: list[float] = []

    regions_total = len(region_embeddings)

    for region, mag_dict in region_embeddings.items():
        if magnifications is not None:
            for mag in requested_set:
                if mag not in mag_dict:
                    coverage_counts[mag] += 1

        available_for_region = [
            mag for mag in sorted(mag_dict) if (not requested_set or mag in requested_set)
        ]

        if len(available_for_region) < 2:
            continue

        averaged_vectors: list[torch.Tensor] = []
        for mag in available_for_region:
            vectors = mag_dict[mag]
            stacked = torch.stack(vectors, dim=0)
            averaged_vectors.append(stacked.mean(dim=0))

        stack = torch.stack(averaged_vectors, dim=0)
        stack = F.normalize(stack, dim=-1)
        centroid_vec = stack.mean(dim=0)
        centroid_norm = torch.norm(centroid_vec)
        if torch.isnan(centroid_norm) or float(centroid_norm.item()) == 0.0:
            continue
        centroid = centroid_vec / centroid_norm

        sims = torch.matmul(stack, centroid)
        mean_sim = sims.mean()
        var = torch.mean((sims - mean_sim) ** 2)

        var_value = float(var.item())
        per_region_variances.append(var_value)
        max_var = _max_variance(len(available_for_region))
        per_region_max_variances.append(max_var)
        if max_var > 0:
            per_region_norms.append(var_value / max_var)
        else:
            per_region_norms.append(0.0)

    regions_used = len(per_region_variances)

    if regions_used == 0:
        missing_info = (
            f" Requested magnifications: {sorted(requested_set)}" if requested_set else ""
        )
        raise ValueError(
            "No regions with at least two of the required magnifications were found to compute MSCI." +
            missing_info
        )

    if magnifications is not None:
        skipped = [mag for mag, count in coverage_counts.items() if count == regions_total]
        if skipped:
            warnings.warn(
                "All regions were missing magnifications "
                f"{skipped}; these levels did not contribute to MSCI.",
                RuntimeWarning,
            )

    variance_tensor = torch.tensor(per_region_variances, dtype=torch.float32)
    mean_variance = float(variance_tensor.mean().item())

    norm_tensor = torch.tensor(per_region_norms, dtype=torch.float32)
    mean_normalised_variance = float(norm_tensor.mean().item()) if len(per_region_norms) else 0.0
    score = max(0.0, min(1.0, 1.0 - mean_normalised_variance))

    representative_max = float(np.mean(per_region_max_variances)) if per_region_max_variances else 0.0

    return MSCIResult(
        score=score,
        mean_variance=mean_variance,
        max_variance=representative_max,
        per_region_variance=per_region_variances,
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
        if msci_config.get("label_column") is not None:
            msci_params["label_column"] = msci_config.get("label_column")
        if msci_config.get("magnifications") is not None:
            msci_params["magnifications"] = msci_config.get("magnifications")

    region_column = msci_params.get("region_column") or _resolve_column(
        "region_column", (train, evald), "region_id"
    )
    magnification_column = msci_params.get("magnification_column") or _resolve_column(
        "magnification_column", (train, evald), "magnification"
    )
    label_column = msci_params.get("label_column") or _resolve_column(
        "label_column", (train, evald), "label"
    )
    msci_params.setdefault("region_column", region_column)
    msci_params.setdefault("magnification_column", magnification_column)
    msci_params.setdefault("label_column", label_column)

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
        feature_payload["region_ids"] = combined_manifest[region_column].astype(str).tolist()
        feature_payload["magnifications"] = combined_manifest[magnification_column].tolist()
        return feature_payload, combined_manifest

    # Attempt to auto-populate manifest information from the feature payloads,
    # preferring the combined (train+eval) set so that regions with different
    # magnifications across splits are still represented.
    combined_features, combined_manifest = _build_msci_payload((train, evald))
    if manifest_source is None and combined_manifest is not None:
        manifest_source = combined_manifest

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
            train,
            evald,
            manifest_source,
            region_column=msci_params.get("region_column", "region_id"),
            magnification_column=msci_params.get("magnification_column", "magnification"),
            label_column=msci_params.get("label_column"),
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
        "TCGA": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/uni_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/multires_txt02.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "label_column": "label",
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/conch_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/multires_txt02.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "label_column": "label",
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/giga_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/multires_txt02.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "label_column": "label",
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/phikon_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/multires_txt02.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "label_column": "label",
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/multires_txt02.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "label_column": "label",
                    "magnifications": [5, 10, 20, 40],
                },
            },
        }
    }
    raw_scores = compute_scores_for_all_datasets(dataset_model_paths, device=args.device)
    ground_truth = {"TCGA": {"uni": 0.94}}
    weights, signs = derive_optimal_weights(raw_scores, ground_truth)
    combined_scores = normalize_and_combine_scores(raw_scores, weights=weights, signs=signs)
    compute_kendall_tau_across_datasets(combined_scores, ground_truth)

    if args.benchmark:
        print("Benchmark:", benchmark_runtime())
