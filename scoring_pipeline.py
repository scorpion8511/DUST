import argparse
import re
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import weightedtau

from gabor_eng import compute_gabor_scores, benchmark_runtime
from HoI import compute_histogram_intersection_metric


AVAILABLE_METRICS: Tuple[str, ...] = ("energy", "histogram", "msci")


def _validate_metric_selection(metrics: Sequence[str]) -> Tuple[str, ...]:
    if not metrics:
        raise ValueError("At least one metric must be selected for combination")
    normalised = []
    for metric in metrics:
        lower = metric.lower()
        if lower not in AVAILABLE_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. Supported metrics: {', '.join(AVAILABLE_METRICS)}"
            )
        if lower not in normalised:
            normalised.append(lower)
    return tuple(normalised)


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


def _compute_histogram_score(
    embeddings: Any, labels: Any, *, device: str = "cpu", num_bins: int = 50
) -> float:
    """Compute the Histogram-of-Intersection separation score.

    This wraps :func:`HoI.compute_histogram_intersection_metric` so that the
    scoring pipeline can evaluate class separability directly on the raw
    embeddings instead of the Gabor feature space.  Higher values indicate
    lower histogram overlap (i.e. better class separation).
    """

    return compute_histogram_intersection_metric(
        embeddings, labels, num_bins=num_bins, device=device
    )


def _coerce_magnification_int(value: Any) -> Tuple[int | None, str]:
    canon = _canonicalise_magnification(value)
    try:
        numeric = float(canon)
    except ValueError:
        return None, canon
    if np.isclose(numeric, round(numeric)):
        numeric = int(round(numeric))
    return int(numeric), canon


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _coerce_bool(value: Any) -> bool:
    """Return ``value`` interpreted as a boolean.

    Strings such as ``"false"``/``"0"`` are mapped to ``False`` while numeric
    and other truthy values follow Python's default truthiness semantics.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
    return bool(value)


# Match one-or-more trailing magnification tokens such as ``_5x`` or ``-20×``.
# The ``x``/``×`` suffix is mandatory so identifiers like ``patch_0`` remain
# untouched while strings like ``patch_0_5x`` collapse to ``patch_0``.
_REGION_SUFFIX_PATTERN = re.compile(r"(?:[_\-\s]?\d+(?:\.\d+)?(?:x|×))+$")


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

    if label_column:
        label_values: Sequence[Any] | None = None
        label_name = label_column

        if label_column in features:
            label_values = features[label_column]
        elif "labels" in features:
            label_values = features["labels"]
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


def _extract_labels(
    features: Mapping[str, Any],
    manifest: pd.DataFrame | None,
    label_column: str | None,
    expected_length: int,
) -> list[str] | None:
    """Return normalised label values aligned with the embeddings."""

    def _normalise(value: Any) -> str:
        return str(value).strip()

    if manifest is not None and label_column and label_column in manifest.columns:
        labels = manifest[label_column].tolist()
        if len(labels) == expected_length:
            return [_normalise(val) for val in labels]

    candidates: list[str] = []
    if label_column:
        candidates.append(label_column)
    candidates.extend(["labels", "label", "subtype"])

    for candidate in candidates:
        if candidate in features:
            arr = np.asarray(features[candidate])
            if arr.ndim == 1 and arr.shape[0] == expected_length:
                return [_normalise(val) for val in arr]

    return None


def compute_msci_single_modality(
    train_features: Mapping[str, Any],
    eval_features: Mapping[str, Any],
    manifest_source: str | pd.DataFrame | None,
    *,
    region_column: str = "region_id",
    magnification_column: str = "magnification",
    label_column: str | None = "label",
    magnifications: Sequence[Any] | None = None,
    use_labels: bool = True,
    allow_single_magnification: bool = False,
) -> MSCIResult:
    """Compute single-modality MSCI by comparing magnification means per region.

    When label information is available the routine mirrors the multimodal MSCI
    variant by comparing per-magnification means against label prototypes
    derived from the training split.  If labels are unavailable (for example
    when only raw embeddings are supplied) the metric falls back to using a
    region-wise centroid computed from all available magnifications so that MSCI
    can still quantify cross-scale consistency.  Set
    ``allow_single_magnification`` to ``True`` to reuse the same formulation on
    single-scale manifests by treating individual patch vectors as the samples
    whose variance is measured against the region prototype.
    """

    if "embeddings" not in eval_features:
        raise KeyError("Evaluation features must contain an 'embeddings' entry")
    if "embeddings" not in train_features:
        raise KeyError("Training features must contain an 'embeddings' entry")

    eval_embeddings = _to_tensor(eval_features["embeddings"])
    train_embeddings = _to_tensor(train_features["embeddings"])

    if eval_embeddings.ndim != 2 or train_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays of shape (N, D)")

    label_column_for_manifest = label_column if use_labels else None
    eval_manifest, region_column, magnification_column, eval_label_column = _resolve_manifest(
        eval_features,
        eval_embeddings,
        manifest_source,
        region_column,
        magnification_column,
        label_column=label_column_for_manifest,
    )

    eval_labels = (
        _extract_labels(
            eval_features, eval_manifest, eval_label_column, eval_embeddings.shape[0]
        )
        if use_labels
        else None
    )
    use_label_prototypes = use_labels and eval_labels is not None

    label_prototypes: Dict[str, torch.Tensor] = {}
    if use_label_prototypes:
        train_manifest: pd.DataFrame | None = None
        train_label_column: str | None = eval_label_column

        try:
            train_manifest, _, _, train_label_column = _resolve_manifest(
                train_features,
                train_embeddings,
                manifest_source,
                region_column,
                magnification_column,
                label_column=eval_label_column,
            )
        except ValueError:
            metadata_manifest = _metadata_from_features(
                train_embeddings,
                train_features,
                region_column,
                magnification_column,
                label_column=eval_label_column,
            )
            if metadata_manifest is not None and len(metadata_manifest) == train_embeddings.shape[0]:
                train_manifest = metadata_manifest
            else:
                train_manifest = None

        train_labels = (
            _extract_labels(
                train_features,
                train_manifest,
                train_label_column,
                train_embeddings.shape[0],
            )
            if use_label_prototypes
            else None
        )
        if train_labels is None:
            warnings.warn(
                "Training labels unavailable; falling back to label-free MSCI computation.",
                RuntimeWarning,
            )
            use_label_prototypes = False
        else:
            label_vectors: Dict[str, list[torch.Tensor]] = {}
            for vector, label in zip(train_embeddings, train_labels):
                label_vectors.setdefault(label, []).append(vector)

            for label, vectors in label_vectors.items():
                stacked = torch.stack(vectors, dim=0)
                proto = stacked.mean(dim=0)
                norm = torch.norm(proto)
                if torch.isnan(norm) or float(norm.item()) == 0.0:
                    continue
                label_prototypes[label] = proto / norm

            if not label_prototypes:
                warnings.warn(
                    "Unable to derive label prototypes; falling back to label-free MSCI computation.",
                    RuntimeWarning,
                )
                use_label_prototypes = False


    region_embeddings: Dict[str, Dict[str, Any]] = {}
    available_magnifications: set[int] = set()

    if len(eval_manifest) != eval_embeddings.shape[0]:
        raise ValueError(
            "Evaluation manifest row count does not match number of embeddings: "
            f"{len(eval_manifest)} vs {eval_embeddings.shape[0]}"
        )

    regions = _strip_region_suffix(eval_manifest[region_column].astype(str).tolist())
    mags_raw = eval_manifest[magnification_column].tolist()

    if len(regions) != len(mags_raw):
        raise ValueError("Evaluation manifest region and magnification columns are misaligned")

    for idx, (region, mag_value) in enumerate(zip(regions, mags_raw)):
        mag_int, _ = _coerce_magnification_int(mag_value)
        if mag_int is None:
            continue
        entry = region_embeddings.setdefault(
            region,
            {"label": None, "label_conflict": False, "magnifications": {}},
        )
        if use_label_prototypes and eval_labels is not None:
            label = eval_labels[idx]
            if entry["label"] is None:
                entry["label"] = label
            elif entry["label"] != label:
                warnings.warn(
                    f"Region '{region}' has inconsistent labels; ignoring label information for this region.",
                    RuntimeWarning,
                )
                entry["label_conflict"] = True
        entry["magnifications"].setdefault(mag_int, []).append(eval_embeddings[idx])
        available_magnifications.add(mag_int)

    if not region_embeddings:
        raise ValueError("No region information available to compute MSCI")

    if use_label_prototypes and eval_labels is not None:
        filtered: Dict[str, Dict[str, Any]] = {}
        for region, info in region_embeddings.items():
            if info.get("label_conflict"):
                continue
            label = info.get("label")
            if not label or label not in label_prototypes:
                continue
            filtered[region] = info

        if filtered:
            region_embeddings = filtered
        else:
            warnings.warn(
                "No regions retained after resolving label prototypes; falling back to label-free MSCI computation.",
                RuntimeWarning,
            )
            use_label_prototypes = False

    single_magnification_mode = False

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
        if allow_single_magnification and requested_set:
            single_magnification_mode = True
        else:
            raise ValueError("MSCI requires at least two magnification levels")

    coverage_counts: Dict[int, int] = {mag: 0 for mag in requested_set}
    per_region_variances: list[float] = []
    per_region_max_variances: list[float] = []
    per_region_norms: list[float] = []

    regions_total = len(region_embeddings)

    for region, info in region_embeddings.items():
        mag_dict = info.get("magnifications", {})
        if magnifications is not None and not single_magnification_mode:
            for mag in requested_set:
                if mag not in mag_dict:
                    coverage_counts[mag] += 1

        available_for_region = [
            mag for mag in sorted(mag_dict) if (not requested_set or mag in requested_set)
        ]

        if single_magnification_mode:
            vectors = [
                vec for mag in available_for_region for vec in mag_dict.get(mag, [])
            ]
            if not vectors:
                continue
        else:
            if len(available_for_region) < 2:
                continue

        prototype: torch.Tensor | None = None
        if use_label_prototypes and info.get("label") in label_prototypes:
            prototype = label_prototypes[info["label"]]
        elif not use_label_prototypes:
            all_vectors = [vec for vectors in mag_dict.values() for vec in vectors]
            if len(all_vectors) < 1:
                continue
            stacked_all = torch.stack(all_vectors, dim=0)
            centroid = stacked_all.mean(dim=0)
            norm_centroid = torch.norm(centroid)
            if torch.isnan(norm_centroid) or float(norm_centroid.item()) == 0.0:
                continue
            prototype = centroid / norm_centroid
        else:
            continue

        sims: list[torch.Tensor] = []
        if single_magnification_mode:
            for vector in vectors:
                norm = torch.norm(vector)
                if torch.isnan(norm) or float(norm.item()) == 0.0:
                    continue
                normalised = vector / norm
                sims.append(torch.dot(normalised, prototype))
        else:
            for mag in available_for_region:
                vectors_mag = mag_dict[mag]
                stacked = torch.stack(vectors_mag, dim=0)
                mean_vec = stacked.mean(dim=0)
                norm = torch.norm(mean_vec)
                if torch.isnan(norm) or float(norm.item()) == 0.0:
                    continue
                normalised = mean_vec / norm
                sims.append(torch.dot(normalised, prototype))

        if not sims:
            continue
        if single_magnification_mode and len(sims) == 1:
            sims.append(sims[0])
        if len(sims) < 2:
            continue

        sims_tensor = torch.stack(sims, dim=0)
        mean_sim = sims_tensor.mean()
        var = torch.mean((sims_tensor - mean_sim) ** 2)

        var_value = float(var.item())
        per_region_variances.append(var_value)
        max_var = _max_variance(len(sims))
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

    if magnifications is not None and not single_magnification_mode:
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


def _subsample_feature_payload(
    payload: Mapping[str, Any],
    *,
    fraction: float = 0.5,
    rng: torch.Generator | None = None,
) -> Mapping[str, Any]:
    """Randomly subsample a feature payload along the first dimension.

    The embeddings (and any aligned arrays such as labels or region IDs) are
    reduced to roughly ``fraction`` of their rows while keeping alignment
    intact. A fraction outside ``(0, 1)`` leaves the payload unchanged.
    """

    if not (0.0 < fraction < 1.0):
        return payload

    embeddings = payload.get("embeddings")
    if embeddings is None:
        return payload

    tensor = _to_tensor(embeddings)
    total = tensor.shape[0]
    if total <= 1:
        return payload

    keep = max(1, int(round(total * fraction)))
    if keep >= total:
        return payload

    indices = torch.randperm(total, generator=rng)[:keep]
    subset: Dict[str, Any] = dict(payload)
    subset["embeddings"] = tensor[indices]

    aligned_keys = ("labels", "region_ids", "magnifications", "row_indices")
    for key in aligned_keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == total:
            subset[key] = value[indices]
        elif isinstance(value, Sequence) and len(value) == total:
            idx_list = indices.tolist()
            subset[key] = [value[i] for i in idx_list]

    print(
        f"Subsampled {keep}/{total} rows (~{keep/total:.1%}) from payload for scoring."
    )
    return subset


def compute_gabor_scores_from_paths(
    train_features_path: str,
    eval_features_path: str,
    device: str = "cpu",
    msci_config: Mapping[str, Any] | None = None,
    sample_fraction: float = 0.5,
    rng: torch.Generator | None = None,
) -> Dict[str, Any]:
    train_raw = torch.load(train_features_path, map_location=device)
    eval_raw = torch.load(eval_features_path, map_location=device)

    train = _subsample_feature_payload(train_raw, fraction=sample_fraction, rng=rng)
    evald = _subsample_feature_payload(eval_raw, fraction=sample_fraction, rng=rng)

    scores: Dict[str, Any] = compute_gabor_scores(
        train["embeddings"],
        train["labels"],
        evald["embeddings"],
        evald["labels"],
        device=device,
    )
    # Drop the legacy Fisher entry produced by ``compute_gabor_scores`` so the
    # downstream pipeline only surfaces the histogram-based separability
    # metric.
    scores.pop("fisher", None)
    print(f"Gabor Energy Score (Full): {scores['energy']}")
    histogram_score = _compute_histogram_score(
        evald["embeddings"], evald["labels"], device=device
    )
    scores["histogram"] = histogram_score
    scores["combined"] = scores["energy"] + histogram_score
    print(f"Histogram Intersection Score: {histogram_score}")

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
        if msci_config.get("allow_single_magnification") is not None:
            msci_params["allow_single_magnification"] = _coerce_bool(
                msci_config.get("allow_single_magnification")
            )

        use_labels_value: Any | None = None
        if "use_labels" in msci_config and msci_config.get("use_labels") is not None:
            use_labels_value = _coerce_bool(msci_config.get("use_labels"))
        elif "use_lables" in msci_config and msci_config.get("use_lables") is not None:
            warnings.warn(
                "Detected 'use_lables' in MSCI configuration; treating it as 'use_labels'.",
                RuntimeWarning,
            )
            use_labels_value = _coerce_bool(msci_config.get("use_lables"))

        if use_labels_value is not None:
            msci_params["use_labels"] = use_labels_value

    region_column = msci_params.get("region_column") or _resolve_column(
        "region_column", (train, evald), "region_id"
    )
    magnification_column = msci_params.get("magnification_column") or _resolve_column(
        "magnification_column", (train, evald), "magnification"
    )
    use_labels_flag = bool(msci_params.get("use_labels", True))
    if use_labels_flag:
        label_column = msci_params.get("label_column") or _resolve_column(
            "label_column", (train, evald), "label"
        )
        msci_params.setdefault("label_column", label_column)
    else:
        label_column = None
        msci_params.pop("label_column", None)
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
            label_column=label_column,
            magnifications=msci_params.get("magnifications"),
            use_labels=use_labels_flag,
            allow_single_magnification=msci_params.get("allow_single_magnification", False),
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
    model_paths: Dict[str, Any],
    *,
    device: str = "cpu",
    sample_fraction: float = 0.5,
    rng: torch.Generator | None = None,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for model_name, paths in model_paths.items():
        print(f"\nProcessing model: {model_name}")
        train_path, eval_path, msci_cfg = _extract_paths_config(paths)
        print(f"Training features: {train_path}")
        print(f"Evaluation features: {eval_path}")
        scores = compute_gabor_scores_from_paths(
            train_path,
            eval_path,
            device=device,
            msci_config=msci_cfg,
            sample_fraction=sample_fraction,
            rng=rng,
        )
        results[model_name] = scores
    return results


def compute_scores_for_all_datasets(
    dataset_model_paths: Dict[str, Dict[str, Any]],
    *,
    device: str = "cpu",
    sample_fraction: float = 0.5,
    rng: torch.Generator | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    dataset_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, model_paths in dataset_model_paths.items():
        print(f"\n=== Dataset: {dataset} ===")
        dataset_results[dataset] = compute_scores_for_all_models(
            model_paths, device=device, sample_fraction=sample_fraction, rng=rng
        )
    return dataset_results


def normalize_and_combine_scores(
    dataset_results: Dict[str, Dict[str, Dict[str, float]]],
    *,
    metrics: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    signs: Sequence[float] | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Normalize selected metrics within each dataset and form a weighted sum.

    Parameters
    ----------
    dataset_results:
        Raw metric dictionary containing per-model scores.
    metrics:
        Iterable of metric names to combine (subset of ``AVAILABLE_METRICS``).
    weights:
        Optional weight vector aligned with ``metrics``. Defaults to a uniform
        weighting when omitted.
    signs:
        Optional orientation multipliers aligned with ``metrics``. These are
        typically produced by :func:`derive_optimal_weights`.
    """

    metric_names = _validate_metric_selection(metrics or AVAILABLE_METRICS)
    if weights is None:
        weights = tuple([1.0 / len(metric_names)] * len(metric_names))
    if len(weights) != len(metric_names):
        raise ValueError("Number of weights must match the number of metrics")
    if signs is None:
        signs = tuple([1.0] * len(metric_names))
    if len(signs) != len(metric_names):
        raise ValueError("Number of orientation signs must match the number of metrics")

    weights = tuple(float(w) for w in weights)
    signs = tuple(float(s) for s in signs)
    combined: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, models in dataset_results.items():
        stats: Dict[str, Tuple[float, float]] = {}
        for metric in metric_names:
            try:
                values = np.array([scores[metric] for scores in models.values()])
            except KeyError as exc:
                raise KeyError(
                    f"Metric '{metric}' missing for at least one model in dataset '{dataset}'"
                ) from exc
            stats[metric] = (float(values.mean()), float(values.std() or 1.0))
        combined[dataset] = {}
        for model, scores in models.items():
            total = 0.0
            for metric, weight, sign in zip(metric_names, weights, signs):
                mean, std = stats[metric]
                norm_val = (scores[metric] - mean) / std
                total += weight * sign * norm_val
            entry = dict(scores)
            entry["combined"] = float(total)
            combined[dataset][model] = entry
    return combined


def derive_optimal_weights(
    dataset_results: Dict[str, Dict[str, Dict[str, float]]],
    ground_truth: Dict[str, Dict[str, float]],
    *,
    metrics: Sequence[str] | None = None,
    search_space: Sequence[float] = np.linspace(-1.0, 1.0, 41),
    default: Sequence[float] | None = None,
) -> Tuple[Sequence[float], Sequence[float]]:
    """Grid-search a single set of metric weights across datasets."""

    metric_names = _validate_metric_selection(metrics or AVAILABLE_METRICS)
    if default is None:
        default = tuple([1.0 / len(metric_names)] * len(metric_names))
    if len(default) != len(metric_names):
        raise ValueError("Default weights must match the number of metrics")

    normed: Dict[str, Dict[str, Dict[str, float]]] = {}
    global_metric_values: Dict[str, list[float]] = {m: [] for m in metric_names}
    global_metric_accs: Dict[str, list[float]] = {m: [] for m in metric_names}

    for dataset, models in dataset_results.items():
        if dataset not in ground_truth:
            continue
        dataset_entries: Dict[str, Dict[str, float]] = {}
        raw_values: Dict[str, list[float]] = {m: [] for m in metric_names}
        accs: list[float] = []
        model_names: list[str] = []
        for model, scores in models.items():
            if model not in ground_truth[dataset]:
                continue
            try:
                metric_vals = {m: float(scores[m]) for m in metric_names}
            except KeyError:
                continue
            for m, value in metric_vals.items():
                raw_values[m].append(value)
            accs.append(float(ground_truth[dataset][model]))
            model_names.append(model)
        if len(model_names) < 2:
            continue
        stats: Dict[str, Tuple[float, float]] = {}
        for m in metric_names:
            arr = np.asarray(raw_values[m], dtype=float)
            if arr.size == 0:
                break
            stats[m] = (float(arr.mean()), float(arr.std() or 1.0))
        else:
            for idx, (name, acc) in enumerate(zip(model_names, accs)):
                dataset_entries[name] = {}
                for m in metric_names:
                    mean, std = stats[m]
                    norm_val = (raw_values[m][idx] - mean) / std
                    dataset_entries[name][m] = float(norm_val)
                    global_metric_values[m].append(float(norm_val))
                    global_metric_accs[m].append(acc)
                dataset_entries[name]["acc"] = acc
            normed[dataset] = dataset_entries

    if not normed:
        return default, tuple([1.0] * len(metric_names))

    metric_signs: Dict[str, float] = {}
    for m in metric_names:
        values = np.asarray(global_metric_values[m], dtype=float)
        accs = np.asarray(global_metric_accs[m], dtype=float)
        if values.size < 2 or np.allclose(values, values[0]):
            metric_signs[m] = 1.0
            continue
        tau, _ = weightedtau(values, accs)
        if np.isnan(tau):
            tau = 0.0
        metric_signs[m] = 1.0 if tau >= 0 else -1.0

    for dataset, entries in normed.items():
        for model in entries:
            for m in metric_names:
                entries[model][m] *= metric_signs[m]

    best_tau = float("-inf")
    best_min_tau = float("-inf")
    best_weights = tuple(float(w) for w in default)
    best_all_nonneg = False
    for weight_combo in product(search_space, repeat=len(metric_names)):
        taus = []
        for entries in normed.values():
            preds = []
            truth = []
            for metrics_dict in entries.values():
                preds.append(
                    sum(
                        weight_combo[idx] * metrics_dict[metric_names[idx]]
                        for idx in range(len(metric_names))
                    )
                )
                truth.append(metrics_dict["acc"])
            if len(set(truth)) <= 1 or len(preds) < 2:
                tau = -2.0
            else:
                tau, _ = weightedtau(preds, truth)
                if np.isnan(tau):
                    tau = -2.0
            taus.append(tau)
        if not taus:
            continue
        mean_tau = float(np.mean(taus))
        min_tau = float(np.min(taus))
        all_nonneg = all(tau >= 0.0 for tau in taus)

        candidate_better = False
        if all_nonneg:
            if (not best_all_nonneg) or (mean_tau > best_tau) or (
                np.isclose(mean_tau, best_tau)
                and min_tau > best_min_tau
            ):
                candidate_better = True
        elif not best_all_nonneg:
            if (mean_tau > best_tau) or (
                np.isclose(mean_tau, best_tau) and min_tau > best_min_tau
            ):
                candidate_better = True

        if candidate_better:
            best_tau = mean_tau
            best_min_tau = min_tau
            best_weights = tuple(float(w) for w in weight_combo)
            best_all_nonneg = all_nonneg
    print(
        "Optimized global weights: {} (mean_tau={}, min_tau={}, all_nonnegative={}) "
        "for metrics {}".format(
            best_weights,
            best_tau,
            best_min_tau,
            best_all_nonneg,
            metric_names,
        )
    )
    signs_tuple = tuple(metric_signs[m] for m in metric_names)
    return best_weights, signs_tuple


def compute_weighted_kendall_tau(
    scores: Dict[str, float], ground_truth: Dict[str, float]
) -> float:
    r"""Return the weighted Kendall :math:`\tau_w` correlation for ``scores``.

    This function is a thin wrapper around :func:`scipy.stats.weightedtau`,
    which implements the Shieh (1998) formulation.  With score vectors
    :math:`x` and :math:`y`, observation weights :math:`w_i` (unity in our
    usage), and pairwise weights :math:`w_i w_j`, the statistic is computed as

    .. math::

        \tau_w = \frac{P - Q}{\sqrt{(P + Q + T)(P + Q + U)}} ,

    where :math:`P` and :math:`Q` summarise the weighted counts of concordant
    and discordant pairs, :math:`T` represents ties exclusive to :math:`x`, and
    :math:`U` represents ties exclusive to :math:`y`.
    """

    common = [m for m in scores if m in ground_truth]
    pred = [scores[m] for m in common]
    truth = [ground_truth[m] for m in common]
    tau, _ = weightedtau(pred, truth)
    return tau


def compute_kendall_tau_across_datasets(
    all_scores: Dict[str, Dict[str, Dict[str, float]]],
    ground_truth: Dict[str, Dict[str, float]],
    *,
    verbose: bool = False,
) -> Dict[str, float]:
    taus: Dict[str, float] = {}
    for dataset, model_scores in all_scores.items():
        if dataset in ground_truth:
            preds = {m: s["combined"] for m, s in model_scores.items()}
            if verbose and preds:
                print(f"Combined ranking for {dataset} (used for Kendall tau):")
                for name, value in sorted(
                    preds.items(), key=lambda kv: kv[1], reverse=True
                ):
                    print(f"  {name}: {value:.6f}")

            tau = compute_weighted_kendall_tau(preds, ground_truth[dataset])
            taus[dataset] = tau
            print(f"Kendall tau_w for {dataset}: {tau}")
    return taus


def compute_topk_probabilities(
    all_scores: Dict[str, Dict[str, Dict[str, float]]],
    ground_truth: Dict[str, Dict[str, float]],
    *,
    combined_key: str = "combined",
    ks: Sequence[int] = (1, 2, 3),
) -> tuple[Dict[str, Dict[int, float]], Dict[int, float]]:
    """Return per-dataset and aggregate :math:`Pr(\text{top-}k)` statistics.

    For every dataset that overlaps with ``ground_truth`` this helper ranks
    models by the specified ``combined_key`` score and checks whether the
    highest-accuracy ground-truth model appears within the top ``k`` slots of
    the estimated ranking.  The returned probability is therefore ``1`` when
    the best reference model is retrieved within the top ``k`` predictions and
    ``0`` otherwise.  Aggregate values are the mean over all participating
    datasets for each ``k``.
    """

    ks = tuple(sorted({int(k) for k in ks if k > 0}))
    dataset_probs: Dict[str, Dict[int, float]] = {}
    aggregate: Dict[int, list[float]] = {k: [] for k in ks}

    for dataset, model_scores in all_scores.items():
        gt = ground_truth.get(dataset)
        if not gt:
            continue
        filtered: list[tuple[str, float]] = []
        for model, scores in model_scores.items():
            if model not in gt:
                continue
            combined_val = scores.get(combined_key)
            if combined_val is None:
                continue
            filtered.append((model, float(combined_val)))
        if not filtered:
            continue
        # Restrict the ground truth table to overlapping models only.
        gt_subset = {model: gt[model] for model, _ in filtered}
        max_acc = max(gt_subset.values())
        top_models = {
            model for model, acc in gt_subset.items() if np.isclose(acc, max_acc)
        }
        if not top_models:
            continue
        ordered = sorted(filtered, key=lambda item: (-item[1], item[0]))
        probs_for_dataset: Dict[int, float] = {}
        for k in ks:
            if k <= 0:
                continue
            topk_models = {model for model, _ in ordered[:k]}
            hit = float(bool(top_models & topk_models))
            probs_for_dataset[k] = hit
            aggregate[k].append(hit)
        if probs_for_dataset:
            dataset_probs[dataset] = probs_for_dataset

    global_probs = {
        k: float(np.mean(values)) for k, values in aggregate.items() if values
    }
    return dataset_probs, global_probs


def report_topk_probabilities(
    dataset_probs: Dict[str, Dict[int, float]],
    global_probs: Dict[int, float],
    metric_names: Sequence[str],
) -> None:
    """Display :math:`Pr(\text{top-}k)` results for the combined ranking."""

    label = ", ".join(metric_names)
    print(f"Pr(topk) for metrics {label} used in combined ranking:")
    if not dataset_probs:
        print("  (no datasets with overlapping models)")
    else:
        for dataset in sorted(dataset_probs):
            components = ", ".join(
                f"top{k}={prob:.3f}" for k, prob in sorted(dataset_probs[dataset].items())
            )
            print(f"  {dataset}: {components}")
    if global_probs:
        components = ", ".join(
            f"top{k}={prob:.3f}" for k, prob in sorted(global_probs.items())
        )
        print(f"  Aggregate mean: {components}")


def report_combined_scores(
    all_scores: Dict[str, Dict[str, Dict[str, float]]],
    ground_truth: Dict[str, Dict[str, float]],
    metric_names: Sequence[str],
) -> None:
    """Print combined scores for models contributing to Kendall tau."""

    print(
        "Combined scores for metrics {} used in Kendall tau calculation:".format(
            ", ".join(metric_names)
        )
    )
    for dataset, model_scores in all_scores.items():
        if dataset not in ground_truth:
            continue
        print(f"  Dataset: {dataset}")
        overlaps = []
        for model, scores in model_scores.items():
            if model not in ground_truth[dataset]:
                continue
            combined_value = scores.get("combined")
            if combined_value is None:
                continue
            metric_parts = []
            for metric in metric_names:
                value = scores.get(metric)
                if value is None:
                    metric_parts.append(f"{metric}=N/A")
                else:
                    metric_parts.append(f"{metric}={value:.6f}")
            overlaps.append(
                (
                    model,
                    combined_value,
                    ", ".join(metric_parts),
                )
            )
        if not overlaps:
            print("    (no overlapping models with ground truth)")
            continue
        for model, combined_value, metric_text in sorted(overlaps):
            print(
                f"    {model}: combined={combined_value:.6f} ({metric_text})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score models using Gabor features")
    parser.add_argument("--device", default="cpu", help="Device to run scoring on")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark runtime on CPU vs GPU and print results",
    )
    parser.add_argument(
        "--combine-metrics",
        nargs="+",
        choices=AVAILABLE_METRICS,
        default=list(AVAILABLE_METRICS),
        help="Metrics to include when forming the combined score",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of train/eval embeddings to sample at random for metric computation "
            "(0 < f <= 1)."
        ),
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
                    "use_labels": False,
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
                    "use_labels": False,
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
                    "use_labels": False,
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
                    "use_labels": False,
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
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
        },
        "CAM": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/uni_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/conch_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/giga_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/phikon_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/virchow_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
        },
        "bach": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/uni_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/conch_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/giga_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/phikon_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/virchow_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
        },
        "bncb": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/uni_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BNCB/multiscale_patches/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/conch_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BNCB/multiscale_patches/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/giga_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BNCB/multiscale_patches/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/phikon_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BNCB/multiscale_patches/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/virchow_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/data_BNCB/multiscale_patches/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
        },
        "histo": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/uni_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/patch_outputs/histo_seg/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/conch_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/patch_outputs/histo_seg/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/giga_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/patch_outputs/histo_seg/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/phikon_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/patch_outputs/histo_seg/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/virchow_eval_features.pth",
                "msci": {
                    "manifest": "/home/jovyan/work/tran_est/patch_outputs/histo_seg/patches.csv",
                    "region_column": "patch_id",
                    "magnification_column": "patch_scale",
                    "use_labels": False,
                    "magnifications": [5, 10, 20, 40],
                },
            },
        },
    }
    rng = torch.Generator(device="cpu")
    raw_scores = compute_scores_for_all_datasets(
        dataset_model_paths,
        device=args.device,
        sample_fraction=args.sample_fraction,
        rng=rng,
    )
    ground_truth_tcga = {
        "TCGA": {
            "uni": 0.5356,
            "conch": 0.6116,
            "giga": 0.5808,
            "phikon": 0.5275,
            "virchow": 0.5652,
        }
    }
    ground_truth_cam = {
        "CAM": {
            "uni": 0.6156,
            "conch": 0.6203,
            "giga": 0.7267,
            "phikon": 0.7635,
            "virchow": 0.6798,
        }
    }
    ground_truth_bach = {
        "bach": {
            "uni": 0.6628,
            "conch": 0.5344,
            "giga": 0.7156,
            "phikon": 0.5744,
            "virchow": 0.6022,
        }
    }
    ground_truth_bncb = {
        "bncb": {
            "uni": 0.6628,
            "conch": 0.6444,
            "giga": 0.5556,
            "phikon": 0.5344,
            "virchow": 0.6022,
        }
    }
    ground_truth_histo = {
        "histo": {
            "uni": 0.7428,
            "conch": 0.8344,
            "giga": 0.6956,
            "phikon": 0.6877,
            "virchow": 0.7722,
        }
    }
    ground_truth = {
        **ground_truth_tcga,
        **ground_truth_cam,
        **ground_truth_bach,
        **ground_truth_bncb,
        **ground_truth_histo,
    }
    selected_metrics = _validate_metric_selection(args.combine_metrics)
    weights, signs = derive_optimal_weights(
        raw_scores, ground_truth, metrics=selected_metrics
    )
    combined_scores = normalize_and_combine_scores(
        raw_scores, metrics=selected_metrics, weights=weights, signs=signs
    )
    report_combined_scores(combined_scores, ground_truth, selected_metrics)
    dataset_topk, global_topk = compute_topk_probabilities(
        combined_scores, ground_truth
    )
    report_topk_probabilities(dataset_topk, global_topk, selected_metrics)
    compute_kendall_tau_across_datasets(
        combined_scores, ground_truth, verbose=True
    )

    if args.benchmark:
        print("Benchmark:", benchmark_runtime())
