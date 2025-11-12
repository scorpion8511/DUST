"""Class-wise UMAP centroids and cross-scale dispersion for unimodal models.

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
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

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

DEFAULT_DATASET_MODEL_PATHS: Dict[str, Dict[str, Mapping[str, object]]] = {
    "TCGA": {
        "uni": {
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
            "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_train_features.pth",
            "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_eval_features.pth",
            "msci": {
                "manifest": "/home/jovyan/work/tran_est/data_TCGA/output/patches.csv",
                "region_column": "patch_id",
                "magnification_column": "patch_scale",
                "label_column": "label",
            },
        },
    }
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


def _extract_samples(
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
    msci_cfg = model_cfg.get("msci") if isinstance(model_cfg, MutableMapping) else None
    if not isinstance(model_cfg, Mapping):
        raise SystemExit("Model configuration must be a mapping with feature paths")

    manifest = _load_manifest(msci_cfg or {}) if msci_cfg else None
    region_column = str(msci_cfg.get("region_column", "region")) if msci_cfg else "region"
    magnification_column = (
        str(msci_cfg.get("magnification_column", "magnification")) if msci_cfg else "magnification"
    )
    label_column = str(msci_cfg.get("label_column")) if msci_cfg and msci_cfg.get("label_column") else None

    payloads: List[SampleBatch] = []
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
        batch = _extract_samples(
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
