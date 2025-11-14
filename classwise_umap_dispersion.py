"""Class-wise UMAP centroids and cross-scale dispersion for foundation models.

This utility loads pre-computed feature archives for a supported dataset/model
pair, projects the embeddings into 2D with UMAP, and summarises how each class
moves across magnifications.  For every class it reports the UMAP centroid at
all available magnifications, the associated covariance ellipse, and the
multi-scale dispersion term

    Delta_c = 2 / (|M| (|M|-1)) * sum_{m<m'} || mu_c(m) - mu_c(m') ||^2,

where ``mu_c(m)`` is the class centroid at magnification ``m``.

The script also aggregates a Fisher-style between/within-class ratio for each
magnification

    Fisher(m) = mean_inter_class_centroid_distance / mean_within_class_scatter,

so the textual summary captures both cross-scale stability (Delta_c) and
per-scale separability (Fisher(m)).  The visualisation restricts itself to
centroids and covariance ellipses to avoid clutter from the full scatter plot.

The defaults target the TCGA unimodal features bundled with the scoring
pipeline, but additional dataset/model mappings can easily be added to the
``DEFAULT_DATASET_MODEL_PATHS`` table below.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

try:
    import umap
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency check
    raise SystemExit(
        "umap-learn is required for dimensionality reduction. Install it via "
        "`pip install umap-learn`."
    ) from exc


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

UNIMODAL = "unimodal"
MULTIMODAL = "multimodal"


DEFAULT_DATASET_MODEL_PATHS: Dict[str, Dict[str, Mapping[str, object]]] = {
    "TCGA": {
        "uni": {
            "type": UNIMODAL,
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/uni_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/uni_eval_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "conch": {
            "type": UNIMODAL,
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/conch_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/conch_eval_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "giga": {
            "type": UNIMODAL,
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/giga_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/giga_eval_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "phikon": {
            "type": UNIMODAL,
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/phikon_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/phikon_eval_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "virchow": {
            "type": UNIMODAL,
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_eval_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
    },
    "TCGA_MULTIMODAL": {
        "plip": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features/plip_features02.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "musk": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features/musk_features02.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "conch": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features/conch_features02.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "pathgen": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features/pathgen_features02.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
    },
    "CAM_MULTIMODAL": {
        "plip": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_cam/plip_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "musk": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_cam/musk_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "conch": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_cam/conch_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "pathgen": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_cam/pathgen_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "biomed": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_cam/biomed_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
    },
    "BACH_MULTIMODAL": {
        "plip": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_bach/plip_features03.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches_paired.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "musk": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_bach/musk_features03.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches_paired.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "conch": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_bach/conch_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches_paired.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "pathgen": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_bach/pathgen_features03.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches_paired.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
        "biomed": {
            "type": MULTIMODAL,
            "path": "/home/jovyan/work/tran_est/MUST/features_multi_bach/biomed_features03.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches_paired.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _canonicalise_magnification(value: object) -> Tuple[float, str]:
    """Return ``(numeric_value, display_label)`` for a magnification token."""

    text = str(value).strip().replace("×", "x")
    if text.lower().endswith("x"):
        text = text[:-1]
    try:
        numeric = float(text)
    except ValueError:
        return math.inf, str(value)
    if math.isclose(numeric, round(numeric)):
        numeric = float(int(round(numeric)))
    return numeric, f"{numeric:g}x"


def _to_numpy(tensor: object) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


@dataclass
class SampleBatch:
    embeddings: np.ndarray
    labels: List[str]
    magnifications: List[str]


def _load_manifest(msci_config: Mapping[str, object]) -> pd.DataFrame | None:
    manifest_path = msci_config.get("manifest") if msci_config else None
    if not manifest_path:
        return None
    path = Path(str(manifest_path)).expanduser()
    if not path.exists():  # pragma: no cover - safety check
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    return df


def _extract_unimodal_samples(
    payload: Mapping[str, object],
    manifest: pd.DataFrame | None,
    region_column: str,
    magnification_column: str,
    label_column: str | None,
) -> SampleBatch:
    embeddings = _to_numpy(payload["embeddings"])  # type: ignore[index]
    num_samples = embeddings.shape[0]

    labels = list(map(str, payload.get("labels", []))) if payload.get("labels") is not None else []
    if labels and len(labels) != num_samples:
        raise ValueError("Mismatch between embeddings and label array lengths")

    magnifications: List[str] = []

    if "magnifications" in payload:
        raw_mags = list(payload["magnifications"])  # type: ignore[index]
        if len(raw_mags) == num_samples:
            magnifications = [_canonicalise_magnification(m)[1] for m in raw_mags]

    if (not magnifications or not labels) and manifest is not None and "row_indices" in payload:
        row_indices = list(payload["row_indices"])  # type: ignore[index]
        if len(row_indices) != num_samples:
            raise ValueError("row_indices length does not match embeddings")
        subset = manifest.iloc[np.asarray(row_indices, dtype=int)]
        if not magnifications:
            magnifications = [
                _canonicalise_magnification(m)[1] for m in subset[magnification_column].tolist()
            ]
        if (not labels or len(labels) != num_samples) and label_column:
            labels = list(map(str, subset[label_column].tolist()))

    if not magnifications:
        raise ValueError("Unable to recover magnification labels for the feature payload")
    if not labels:
        raise ValueError("Unable to recover class labels for the feature payload")

    return SampleBatch(embeddings=embeddings, labels=labels, magnifications=magnifications)


def _identifier_variants(value: str) -> List[str]:
    tokens: Dict[str, None] = {}

    def _add(token: Optional[str]) -> None:
        if token is None:
            return
        text = token.strip()
        if not text:
            return
        tokens.setdefault(text, None)
        tokens.setdefault(text.lower(), None)

    _add(value)
    if "::" in value:
        _add(value.split("::", 1)[1])
    base = os.path.basename(value)
    _add(base)
    if "::" in base:
        _add(base.split("::", 1)[1])
    root, _ = os.path.splitext(base)
    _add(root)
    if "::" in root:
        _add(root.split("::", 1)[1])
    segment = root
    while "_" in segment:
        segment = segment.rsplit("_", 1)[0]
        _add(segment)
    return list(tokens.keys())


def _build_manifest_lookup(
    manifest: pd.DataFrame,
    region_column: str,
    label_column: str,
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    ambiguous: set[str] = set()

    region_series = manifest[region_column].astype(str)
    label_series = manifest[label_column].astype(str)

    for region_value, label_value in zip(region_series, label_series, strict=False):
        variants = _identifier_variants(region_value)
        clean_label = label_value.strip()
        for token in variants:
            if token in ambiguous:
                continue
            if token in lookup:
                if lookup[token] != clean_label:
                    ambiguous.add(token)
                    lookup.pop(token, None)
                continue
            lookup[token] = clean_label

    if ambiguous:
        warnings.warn(
            "Some manifest identifiers map to multiple labels; those entries were ignored during lookup.",
            RuntimeWarning,
            stacklevel=2,
        )

    return lookup


def _resolve_region_labels(
    metadata: Mapping[str, object],
    manifest: Optional[pd.DataFrame],
    region_column: str,
    label_column: Optional[str],
) -> List[str]:
    region_ids = list(map(str, metadata.get("region_ids", [])))
    if not region_ids:
        raise ValueError("Multimodal feature archive is missing region identifiers in metadata")

    raw_labels = metadata.get("labels")
    if isinstance(raw_labels, Sequence) and len(raw_labels) == len(region_ids):
        processed = [str(value).strip() for value in raw_labels]
        if len({label for label in processed if label}) >= 1:
            return processed

    if manifest is None:
        raise ValueError("Manifest is required to resolve class labels for multimodal features")

    csv_mapping = metadata.get("csv_region_mapping", {})
    if not isinstance(csv_mapping, Mapping):
        csv_mapping = {}

    candidate_columns: List[str] = []
    if label_column and label_column in manifest.columns:
        candidate_columns.append(label_column)
    for fallback in ("label", "labels", "subtype"):
        if fallback in manifest.columns and fallback not in candidate_columns:
            candidate_columns.append(fallback)

    if not candidate_columns:
        raise ValueError("Manifest does not contain any suitable label columns")

    for column in candidate_columns:
        lookup = _build_manifest_lookup(manifest, region_column, column)
        resolved: List[str] = []
        missing: List[str] = []
        for region_id in region_ids:
            candidates = _identifier_variants(region_id)
            if region_id in csv_mapping:
                for item in csv_mapping[region_id]:
                    candidates.extend(_identifier_variants(str(item)))
            seen: Dict[str, None] = {}
            label_value: Optional[str] = None
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen[candidate] = None
                if candidate in lookup:
                    label_value = lookup[candidate]
                    break
            if label_value is None:
                missing.append(region_id)
                break
            resolved.append(label_value)
        if resolved and len(resolved) == len(region_ids):
            return resolved
        if missing:
            warnings.warn(
                (
                    f"Unable to resolve labels for regions {missing[:5]} using column '{column}'. "
                    "Trying fallback columns if available."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

    raise ValueError("Failed to resolve class labels for the multimodal features")


def _extract_multimodal_samples(
    payload: Mapping[str, object],
    manifest: Optional[pd.DataFrame],
    region_column: str,
    label_column: Optional[str],
) -> SampleBatch:
    if "image_embeddings" not in payload:
        raise ValueError("Multimodal feature archive does not contain 'image_embeddings'")

    raw_images = payload["image_embeddings"]
    if not isinstance(raw_images, Mapping):
        raise TypeError("Expected a mapping of magnification to image embeddings")

    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    if not isinstance(metadata, Mapping):
        metadata = {}

    labels = _resolve_region_labels(metadata, manifest, region_column, label_column)

    image_embeddings: Dict[float, np.ndarray] = {}
    for key, value in raw_images.items():
        try:
            mag = float(key)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid magnification key in image embeddings: {key!r}") from None
        image_embeddings[mag] = _to_numpy(value)

    if not image_embeddings:
        raise ValueError("No image embeddings were found in the multimodal archive")

    ordered_mags = sorted(
        image_embeddings.keys(), key=lambda mag: _canonicalise_magnification(mag)[0]
    )
    ordered_labels = []
    ordered_magnifications: List[str] = []
    embedding_blocks: List[np.ndarray] = []

    num_regions = len(labels)
    for mag in ordered_mags:
        tensor = image_embeddings[mag]
        if tensor.ndim != 2:
            raise ValueError(f"Embeddings for magnification {mag} must be rank-2 tensors")
        if tensor.shape[0] != num_regions:
            raise ValueError(
                f"Magnification {mag} contains {tensor.shape[0]} embeddings but metadata lists {num_regions} regions"
            )
        embedding_blocks.append(tensor)
        label_copy = [str(label) for label in labels]
        ordered_labels.extend(label_copy)
        _, label_mag = _canonicalise_magnification(mag)
        ordered_magnifications.extend([label_mag] * num_regions)

    embeddings = np.concatenate(embedding_blocks, axis=0).astype(np.float32)
    return SampleBatch(embeddings=embeddings, labels=ordered_labels, magnifications=ordered_magnifications)


def _concatenate_batches(batches: Iterable[SampleBatch]) -> SampleBatch:
    embeddings_list: List[np.ndarray] = []
    labels: List[str] = []
    mags: List[str] = []
    for batch in batches:
        embeddings_list.append(batch.embeddings)
        labels.extend(batch.labels)
        mags.extend(batch.magnifications)
    return SampleBatch(
        embeddings=np.concatenate(embeddings_list, axis=0),
        labels=labels,
        magnifications=mags,
    )


@dataclass
class ClassMagnificationStat:
    centroid: np.ndarray
    covariance: np.ndarray
    count: int


@dataclass
class ClassSummary:
    stats: Dict[str, ClassMagnificationStat]
    delta: float


def compute_class_summaries(
    coords: np.ndarray,
    labels: Sequence[str],
    magnifications: Sequence[str],
) -> Dict[str, ClassSummary]:
    classes = sorted(set(labels))
    _, mag_labels = zip(*sorted({_canonicalise_magnification(m) for m in magnifications}))
    summaries: Dict[str, ClassSummary] = {}

    for cls in classes:
        stats: Dict[str, ClassMagnificationStat] = {}
        for mag in mag_labels:
            idx = [i for i, (c, m) in enumerate(zip(labels, magnifications)) if c == cls and m == mag]
            if not idx:
                continue
            subset = coords[idx, :]
            centroid = subset.mean(axis=0)
            if subset.shape[0] > 1:
                covariance = np.cov(subset, rowvar=False)
            else:
                covariance = np.eye(coords.shape[1]) * 1e-6
            stats[mag] = ClassMagnificationStat(centroid=centroid, covariance=covariance, count=len(idx))

        mags_for_class = sorted(stats.keys(), key=lambda m: _canonicalise_magnification(m)[0])
        num_mags = len(mags_for_class)
        if num_mags >= 2:
            pairwise = 0.0
            for mag_a, mag_b in combinations(mags_for_class, 2):
                diff = stats[mag_a].centroid - stats[mag_b].centroid
                pairwise += float(np.dot(diff, diff))
            delta = 2.0 * pairwise / (num_mags * (num_mags - 1))
        else:
            delta = 0.0

        summaries[cls] = ClassSummary(stats=stats, delta=delta)

    return summaries


def compute_fisher_statistics(
    summaries: Mapping[str, ClassSummary]
) -> Dict[str, Dict[str, float]]:
    """Return Fisher-style between/within ratios per magnification.

    For each magnification the function gathers the available class centroids
    and covariance estimates, computes the mean pairwise centroid distance, and
    divides it by the mean within-class scatter (trace of the covariance).
    """

    magnifications = sorted(
        {
            mag
            for summary in summaries.values()
            for mag in summary.stats.keys()
        },
        key=lambda m: _canonicalise_magnification(m)[0],
    )

    fisher: Dict[str, Dict[str, float]] = {}

    for mag in magnifications:
        centroids: List[np.ndarray] = []
        scatters: List[Tuple[float, int]] = []

        for summary in summaries.values():
            stat = summary.stats.get(mag)
            if not stat:
                continue
            centroids.append(stat.centroid)
            scatter = float(np.trace(stat.covariance))
            scatters.append((scatter, stat.count))

        if not centroids:
            continue

        if len(centroids) >= 2:
            total = 0.0
            num_pairs = 0
            for a, b in combinations(range(len(centroids)), 2):
                dist = float(np.linalg.norm(centroids[a] - centroids[b]))
                total += dist
                num_pairs += 1
            mean_inter = total / num_pairs if num_pairs else float("nan")
        else:
            mean_inter = float("nan")

        if scatters:
            total_scatter = sum(value * count for value, count in scatters)
            total_count = sum(count for _, count in scatters)
            mean_within = total_scatter / total_count if total_count else float("nan")
        else:
            mean_within = float("nan")

        if math.isnan(mean_within) or math.isclose(mean_within, 0.0):
            ratio = float("nan")
        else:
            ratio = mean_inter / mean_within

        fisher[mag] = {
            "mean_inter": mean_inter,
            "mean_within": mean_within,
            "ratio": ratio,
        }

    return fisher


def _plot_class_dispersion(
    coords: np.ndarray,
    labels: Sequence[str],
    magnifications: Sequence[str],
    summaries: Mapping[str, ClassSummary],
    output_path: Path,
) -> None:
    classes = sorted(summaries.keys())
    _, mag_labels = zip(*sorted({_canonicalise_magnification(m) for m in magnifications}))

    colors = plt.cm.get_cmap("tab10", len(classes))
    markers = ["o", "s", "^", "D", "P", "X", "*", "v"]
    marker_map = {
        mag: markers[i % len(markers)] for i, mag in enumerate(mag_labels)
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for class_index, cls in enumerate(classes):
        colour = colors(class_index)

        for mag in mag_labels:
            stat = summaries[cls].stats.get(mag)
            if not stat:
                continue
            centroid = stat.centroid
            cov = stat.covariance
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals = np.maximum(vals[order], 1e-8)
            vecs = vecs[:, order]
            angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2.0 * np.sqrt(vals)
            ellipse = Ellipse(
                xy=centroid,
                width=width,
                height=height,
                angle=angle,
                facecolor="none",
                edgecolor=colour,
                lw=2,
                alpha=0.8,
            )
            ax.add_patch(ellipse)
            ax.scatter(
                centroid[0],
                centroid[1],
                color=colour,
                marker=marker_map[mag],
                s=80,
                label=None,
            )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=cls, markerfacecolor=colors(i), markersize=10)
        for i, cls in enumerate(classes)
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=marker_map[mag], color="k", linestyle="", label=mag)
        for mag in mag_labels
    ]
    legend1 = ax.legend(handles=handles, title="Class", loc="upper right")
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, title="Magnification", loc="lower right")

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Class-wise UMAP centroids with covariance ellipses")
    ax.grid(True, linestyle=":", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved UMAP dispersion figure to {output_path}")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="TCGA",
        choices=sorted(DEFAULT_DATASET_MODEL_PATHS.keys()),
        help="Dataset key to load embeddings from.",
    )
    parser.add_argument(
        "--model",
        default="uni",
        help="Model key within the selected dataset configuration.",
    )
    parser.add_argument(
        "--split",
        default="both",
        choices=["train", "eval", "both"],
        help="Which feature split(s) to include when building the UMAP embedding.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbours for UMAP (controls local/global balance).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP minimum distance hyperparameter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("classwise_umap_dispersion.png"),
        help="Path to save the UMAP scatter/ellipse figure.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for UMAP initialisation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_config = DEFAULT_DATASET_MODEL_PATHS.get(args.dataset, {})
    if args.model not in dataset_config:
        raise SystemExit(
            f"Model '{args.model}' is not configured for dataset '{args.dataset}'."
        )

    model_cfg = dataset_config[args.model]
    if not isinstance(model_cfg, Mapping):
        raise SystemExit("Model configuration must be a mapping with feature paths")

    model_type = str(model_cfg.get("type", UNIMODAL))
    msci_cfg = model_cfg.get("msci") if isinstance(model_cfg, MutableMapping) else None

    manifest = _load_manifest(msci_cfg or {}) if msci_cfg else None
    region_column = str(msci_cfg.get("region_column", "region")) if msci_cfg else "region"
    magnification_column = (
        str(msci_cfg.get("magnification_column", "magnification")) if msci_cfg else "magnification"
    )
    label_column = str(msci_cfg.get("label_column")) if msci_cfg and msci_cfg.get("label_column") else None

    payloads: List[SampleBatch] = []
    if model_type == MULTIMODAL:
        path_map: Dict[str, object] = {}
        for split_key in ("train", "eval"):
            if split_key in model_cfg:
                path_map[split_key] = model_cfg[split_key]
        if "path" in model_cfg:
            path_map.setdefault("eval", model_cfg["path"])

        if not path_map:
            raise SystemExit("Multimodal configuration must define at least one feature path")

        if args.split == "both":
            splits = tuple(path_map.keys())
        else:
            splits = (args.split,) if args.split in path_map else ()
            if not splits:
                splits = tuple(path_map.keys())

        for split in splits:
            path = path_map.get(split)
            if not path:
                continue
            feature_path = Path(str(path)).expanduser()
            if not feature_path.exists():  # pragma: no cover - safety check
                raise FileNotFoundError(f"Feature archive not found: {feature_path}")
            payload = torch.load(feature_path, map_location="cpu")
            batch = _extract_multimodal_samples(
                payload,
                manifest,
                region_column=region_column,
                label_column=label_column,
            )
            payloads.append(batch)
    else:
        splits: Sequence[str]
        if args.split == "both":
            splits = ("train", "eval")
        else:
            splits = (args.split,)

        for split in splits:
            path = model_cfg.get(split)
            if not path:
                continue
            feature_path = Path(str(path)).expanduser()
            if not feature_path.exists():  # pragma: no cover - safety check
                raise FileNotFoundError(f"Feature archive not found: {feature_path}")
            payload = torch.load(feature_path, map_location="cpu")
            if "embeddings" not in payload:
                raise ValueError(f"Feature archive '{feature_path}' does not contain embeddings")
            batch = _extract_unimodal_samples(
                payload,
                manifest,
                region_column=region_column,
                magnification_column=magnification_column,
                label_column=label_column,
            )
            payloads.append(batch)

    if not payloads:
        raise SystemExit("No feature payloads were loaded. Check the configuration paths.")

    combined = _concatenate_batches(payloads)
    embeddings = combined.embeddings.astype(np.float32)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        metric="euclidean",
    )
    coords = reducer.fit_transform(embeddings)

    summaries = compute_class_summaries(coords, combined.labels, combined.magnifications)

    print("Class-wise centroid dispersion summary:")
    for cls, summary in summaries.items():
        mags_sorted = sorted(
            summary.stats.keys(), key=lambda m: _canonicalise_magnification(m)[0]
        )
        print(f"- Class {cls}: |M|={len(mags_sorted)}, Delta_c={summary.delta:.6f}")
        for mag in mags_sorted:
            stat = summary.stats[mag]
            cx, cy = stat.centroid
            print(
                f"    • {mag}: n={stat.count}, centroid=({cx:.4f}, {cy:.4f})"
            )

    fisher_stats = compute_fisher_statistics(summaries)
    if fisher_stats:
        print("\nFisher separability by magnification:")
        for mag in sorted(
            fisher_stats.keys(), key=lambda m: _canonicalise_magnification(m)[0]
        ):
            metrics = fisher_stats[mag]
            mean_inter = metrics["mean_inter"]
            mean_within = metrics["mean_within"]
            ratio = metrics["ratio"]
            print(
                f"- {mag}: mean_inter={mean_inter:.4f}, "
                f"mean_within={mean_within:.4f}, Fisher={ratio:.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_class_dispersion(
        coords,
        combined.labels,
        combined.magnifications,
        summaries,
        args.output,
    )
    print(f"Saved UMAP dispersion figure to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
