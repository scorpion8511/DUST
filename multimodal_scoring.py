"""Utilities for evaluating multimodal transferability scores.

This module implements two metrics:

* Magnification-Scale Consistency Index (MSCI)
* Cross-Modal Mutual Information lower-bound (CMI-LB)

Both metrics operate on paired image/text embeddings stored in a `.pth`
file.  The expected format is a dictionary containing a mapping from
magnification levels to image embeddings and a tensor of text embeddings::

    {
        "image_embeddings": {
            5: torch.Tensor [N, D],
            10: torch.Tensor [N, D],
            20: torch.Tensor [N, D],
        },
        "text_embeddings": torch.Tensor [N, D]
    }

Embeddings are L2-normalised prior to computing similarities.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import weightedtau


DEFAULT_MAGNIFICATIONS: Sequence[int] = (5, 10, 20)

DEFAULT_GROUND_TRUTH: Dict[str, Dict[str, float]] = {
    "TCGA": {
        "plip": 0.59,
        "musk": 0.65,
        "conch": 0.58,
        "pathgen": 0.60,
    },
    "CAM": {
        "plip": 0.6222,
        "musk": 0.6434,
        "conch": 0.5889,
        "pathgen": 0.5667,
        "biomed": 0.5444,
    },
}

DEFAULT_DATASET_MODEL_PATHS: Dict[str, Dict[str, str]] = {
    "TCGA": {
        "plip": "/home/jovyan/work/tran_est/MUST/features/plip_features02.pth",
        "musk": "/home/jovyan/work/tran_est/MUST/features/musk_features02.pth",
        "conch": "/home/jovyan/work/tran_est/MUST/features/conch_features02.pth",
        "pathgen": "/home/jovyan/work/tran_est/MUST/features/pathgen_features02.pth",
    },
    "CAM": {
        "plip": "/home/jovyan/work/tran_est/MUST/features_multi_cam/plip_features.pth",
        "musk": "/home/jovyan/work/tran_est/MUST/features_multi_cam/musk_features.pth",
        "conch": "/home/jovyan/work/tran_est/MUST/features_multi_cam/conch_features.pth",
        "pathgen": "/home/jovyan/work/tran_est/MUST/features_multi_cam/pathgen_features.pth",
        "biomed": "/home/jovyan/work/tran_est/MUST/features_multi_cam/biomed_features.pth",
    },
}


@dataclass
class MSCIResult:
    """Container describing the MSCI metric."""

    msci: float
    mean_variance: float
    max_variance: float
    per_region_variance: torch.Tensor
    similarities: torch.Tensor
    magnifications: Sequence[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "msci": float(self.msci),
            "mean_variance": float(self.mean_variance),
            "max_variance": float(self.max_variance),
            "per_region_variance": self.per_region_variance.tolist(),
            "magnifications": list(self.magnifications),
        }


@dataclass
class CMILBResult:
    """Container describing the CMI-LB metric for a single magnification."""

    temperature: float
    loss_x: float
    loss_y: float
    cmi_lb: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "temperature": float(self.temperature),
            "loss_x": float(self.loss_x),
            "loss_y": float(self.loss_y),
            "cmi_lb": float(self.cmi_lb),
        }


def _ensure_magnification_dict(
    image_embeddings: Mapping[int, torch.Tensor], device: torch.device
) -> Dict[int, torch.Tensor]:
    result: Dict[int, torch.Tensor] = {}
    for key, value in image_embeddings.items():
        try:
            mag = int(key)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid magnification key: {key!r}") from exc
        result[mag] = torch.as_tensor(value, device=device, dtype=torch.float32)
    return result


def load_multimodal_embeddings(
    path: str, device: torch.device
) -> tuple[Dict[int, torch.Tensor], torch.Tensor, Mapping[str, object], Optional[Sequence[int]]]:
    """Load multimodal embeddings from ``path``.

    Parameters
    ----------
    path:
        Torch ``.pth`` file containing image and text embeddings.
    device:
        Device on which tensors should be allocated.
    """

    data: MutableMapping[str, object] = torch.load(path, map_location=device)
    if "image_embeddings" not in data or "text_embeddings" not in data:
        raise KeyError(
            "Expected keys 'image_embeddings' and 'text_embeddings' in the feature file."
        )

    images = _ensure_magnification_dict(data["image_embeddings"], device)
    texts = torch.as_tensor(data["text_embeddings"], device=device, dtype=torch.float32)

    metadata = data.get("metadata") if isinstance(data, Mapping) else None
    if not isinstance(metadata, Mapping):
        metadata = {}

    stored_magnifications: Optional[Sequence[int]] = None
    raw_mags = data.get("magnifications") if isinstance(data, Mapping) else None
    if isinstance(raw_mags, Sequence):
        try:
            stored_magnifications = [int(mag) for mag in raw_mags]
        except (TypeError, ValueError):
            warnings.warn(
                "Encountered non-integer magnification identifiers in metadata; falling back to detected keys.",
                RuntimeWarning,
                stacklevel=2,
            )
            stored_magnifications = None

    return images, texts, metadata, stored_magnifications


def _normalise_patch_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    base = os.path.basename(str(value))
    root, _ = os.path.splitext(base)
    return root or base


def _candidate_region_tokens(region_id: str, csv_map: Mapping[object, Sequence[object]]) -> Sequence[str]:
    candidates = {str(region_id)}
    for key, values in csv_map.items():
        if str(key) == str(region_id):
            for item in values:
                candidates.add(str(item))
    return list(candidates)


def _validate_alignment(
    metadata: Mapping[str, object],
    image_embeddings: Mapping[int, torch.Tensor],
    text_embeddings: torch.Tensor,
) -> None:
    region_ids = metadata.get("region_ids") if metadata else None
    patches = metadata.get("patches") if metadata else None
    csv_map = metadata.get("csv_region_mapping") if metadata else None

    if not isinstance(region_ids, Sequence):
        return

    if text_embeddings.shape[0] != len(region_ids):
        warnings.warn(
            "Metadata region count does not match embedding rows; skipping alignment validation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    patch_lookup: Dict[int, Sequence[Optional[str]]] = {}
    if isinstance(patches, Mapping):
        for key, value in patches.items():
            try:
                mag = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, Sequence):
                patch_lookup[mag] = [
                    _normalise_patch_name(item) if item is not None else None for item in value
                ]

    csv_mapping = csv_map if isinstance(csv_map, Mapping) else {}

    for mag, tensor in image_embeddings.items():
        if tensor.shape[0] != len(region_ids):
            warnings.warn(
                f"Magnification {mag} has {tensor.shape[0]} embeddings but metadata lists {len(region_ids)} regions;"
                " skipping patch-name validation for this magnification.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        patches_for_mag = patch_lookup.get(mag)
        if not patches_for_mag:
            continue

        if len(patches_for_mag) != len(region_ids):
            warnings.warn(
                f"Magnification {mag} patch metadata has length {len(patches_for_mag)} but expected {len(region_ids)};"
                " skipping validation for this magnification.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        for idx, region_id in enumerate(region_ids):
            patch_name = patches_for_mag[idx]
            if patch_name is None:
                continue
            candidates = _candidate_region_tokens(str(region_id), csv_mapping if isinstance(csv_mapping, Mapping) else {})
            if not any(token and token in patch_name for token in candidates):
                warnings.warn(
                    f"Patch '{patch_name}' (magnification {mag}) does not appear to match region '{region_id}'.",
                    RuntimeWarning,
                    stacklevel=2,
                )


def _max_variance(num_magnifications: int) -> float:
    """Return the theoretical maximum variance for similarities in [-1, 1]."""

    if num_magnifications <= 0:
        raise ValueError("num_magnifications must be positive.")

    # The maximum variance occurs when the similarities are split between the
    # extreme values -1 and +1. Iterate over all possible splits to find the
    # tightest upper bound for the provided number of magnifications.
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


def compute_msci(
    image_embeddings: Mapping[int, torch.Tensor],
    text_embeddings: torch.Tensor,
    magnifications: Iterable[int] = DEFAULT_MAGNIFICATIONS,
) -> MSCIResult:
    """Compute the Magnification-Scale Consistency Index (MSCI).

    Parameters
    ----------
    image_embeddings:
        Mapping from magnification level to image embedding tensor with
        shape ``[N, D]``. All tensors must share the same number of rows.
    text_embeddings:
        Tensor of text embeddings with shape ``[N, D]``.
    magnifications:
        Ordered iterable of magnification levels to include. Missing
        magnifications raise a ``KeyError``.
    """

    mags = list(magnifications)
    if not mags:
        raise ValueError("At least one magnification is required to compute MSCI.")

    text_norm = F.normalize(text_embeddings, dim=-1)
    similarities = []
    for mag in mags:
        if mag not in image_embeddings:
            raise KeyError(f"Missing embeddings for magnification {mag}.")
        img = F.normalize(image_embeddings[mag], dim=-1)
        if img.shape != text_norm.shape:
            raise ValueError(
                f"Image and text embeddings must have matching shapes, got {img.shape} vs {text_norm.shape}."
            )
        similarities.append(torch.sum(img * text_norm, dim=-1))

    sim_tensor = torch.stack(similarities, dim=-1)
    mean_sim = sim_tensor.mean(dim=-1, keepdim=True)
    variance = torch.mean((sim_tensor - mean_sim) ** 2, dim=-1)
    mean_variance = variance.mean().item()

    max_var = _max_variance(len(mags))
    if max_var <= 0:
        msci_score = 0.0
    else:
        msci_score = 1.0 - (mean_variance / max_var)
        msci_score = float(max(0.0, min(1.0, msci_score)))

    return MSCIResult(
        msci=msci_score,
        mean_variance=mean_variance,
        max_variance=max_var,
        per_region_variance=variance,
        similarities=sim_tensor,
        magnifications=mags,
    )


def _validate_embeddings(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            f"Image and text embeddings must have identical shapes, got {image_embeddings.shape} vs {text_embeddings.shape}."
        )
    if image_embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be rank-2 tensors of shape [N, D], received shape {image_embeddings.shape}."
        )
    if image_embeddings.shape[0] == 0:
        raise ValueError("At least one embedding pair is required to compute CMI-LB.")

    x = F.normalize(image_embeddings, dim=-1)
    y = F.normalize(text_embeddings, dim=-1)
    return x, y


def _estimate_temperature(
    similarities: torch.Tensor,
    min_temperature: float,
    max_temperature: float,
    steps: int,
) -> float:
    """Search for the temperature that minimises the symmetric InfoNCE loss."""

    if steps <= 0:
        raise ValueError("Temperature search requires a positive number of steps.")
    if not (min_temperature > 0 and max_temperature > 0):
        raise ValueError("Temperature bounds must be strictly positive.")
    if min_temperature >= max_temperature:
        raise ValueError("min_temperature must be smaller than max_temperature.")

    log_min = math.log10(min_temperature)
    log_max = math.log10(max_temperature)
    temperatures = torch.logspace(log_min, log_max, steps, device=similarities.device, dtype=similarities.dtype)

    sims = similarities.unsqueeze(0) / temperatures.view(-1, 1, 1)

    log_prob_x = sims.log_softmax(dim=-1)
    log_prob_y = sims.transpose(-1, -2).log_softmax(dim=-1)

    diag_x = torch.diagonal(log_prob_x, dim1=-2, dim2=-1)
    diag_y = torch.diagonal(log_prob_y, dim1=-2, dim2=-1)

    loss_x = -diag_x.mean(dim=-1)
    loss_y = -diag_y.mean(dim=-1)
    symmetric_loss = 0.5 * (loss_x + loss_y)

    best_index = torch.argmin(symmetric_loss).item()
    return float(temperatures[best_index].item())


def compute_cmi_lb(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float | None,
    *,
    min_temperature: float,
    max_temperature: float,
    temperature_steps: int,
) -> CMILBResult:
    """Compute the CMI-LB metric for paired embeddings."""

    x, y = _validate_embeddings(image_embeddings, text_embeddings)

    similarities = torch.matmul(x, y.T)

    if temperature is None:
        temperature = _estimate_temperature(
            similarities, min_temperature=min_temperature, max_temperature=max_temperature, steps=temperature_steps
        )
    elif temperature <= 0:
        raise ValueError("Temperature must be positive for CMI-LB computation.")

    logits = similarities / temperature
    n = logits.shape[0]

    labels = torch.arange(n, device=logits.device)
    log_prob_x = F.log_softmax(logits, dim=-1)
    log_prob_y = F.log_softmax(logits.T, dim=-1)
    loss_x = -log_prob_x[labels, labels].mean()
    loss_y = -log_prob_y[labels, labels].mean()

    log_n = math.log(n)
    cmi_lb = 0.5 * ((log_n - loss_x.item()) + (log_n - loss_y.item()))
    return CMILBResult(temperature=float(temperature), loss_x=float(loss_x.item()), loss_y=float(loss_y.item()), cmi_lb=float(cmi_lb))


def compute_cmi_lb_across_magnifications(
    image_embeddings: Mapping[int, torch.Tensor],
    text_embeddings: torch.Tensor,
    magnifications: Iterable[int],
    temperature: float | None,
    *,
    min_temperature: float,
    max_temperature: float,
    temperature_steps: int,
) -> Dict[int, CMILBResult]:
    """Compute CMI-LB for each magnification individually."""

    results: Dict[int, CMILBResult] = {}
    for mag in magnifications:
        if mag not in image_embeddings:
            raise KeyError(f"Missing embeddings for magnification {mag}.")
        results[mag] = compute_cmi_lb(
            image_embeddings[mag],
            text_embeddings,
            temperature,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            temperature_steps=temperature_steps,
        )
    return results


def _summarise_for_json(msci_result: MSCIResult, cmi_results: Dict[int, CMILBResult], cmi_avg: float) -> Dict[str, object]:
    """Convert metric objects into a JSON-serialisable dictionary."""

    return {
        "msci": msci_result.to_dict(),
        "cmi_lb": {mag: result.to_dict() for mag, result in cmi_results.items()},
        "cmi_lb_mean": cmi_avg,
    }


def _compute_weighted_kendall_tau(
    scores: Dict[str, float], ground_truth: Dict[str, float]
) -> Optional[float]:
    """Compute weighted Kendall tau between predictions and ground truth."""

    overlap = [name for name in scores if name in ground_truth]
    if len(overlap) < 2:
        return None

    predictions = [scores[name] for name in overlap]
    truth = [ground_truth[name] for name in overlap]
    tau, _ = weightedtau(predictions, truth)
    return float(tau)


def _load_ground_truth(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return DEFAULT_GROUND_TRUTH

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {
        str(dataset): {str(model): float(score) for model, score in models.items()}
        for dataset, models in data.items()
    }


def _normalise_dataset_argument(
    dataset: Optional[Sequence[str] | str],
) -> Tuple[str, ...]:
    """Normalise dataset CLI arguments into a tuple of dataset identifiers."""

    if dataset is None:
        return tuple(DEFAULT_GROUND_TRUTH.keys())

    if isinstance(dataset, str):
        parts = [item.strip() for item in dataset.split(",") if item.strip()]
        return tuple(parts) if parts else tuple(DEFAULT_GROUND_TRUTH.keys())

    normalised: list[str] = []
    for item in dataset:
        if item is None:
            continue
        for part in str(item).split(","):
            part = part.strip()
            if part and part not in normalised:
                normalised.append(part)

    if not normalised:
        return tuple(DEFAULT_GROUND_TRUTH.keys())

    return tuple(normalised)


def _optimise_combined_scores(
    per_dataset_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    dataset_truths: Mapping[str, Mapping[str, float]],
    search_space: Sequence[float],
) -> Optional[
    Tuple[Tuple[float, float], Dict[str, float], Dict[str, Dict[str, float]]]
]:
    """Search for shared MSCI/CMI-LB weights across multiple datasets."""

    if not per_dataset_scores or not dataset_truths:
        return None

    prepared: Dict[str, Dict[str, object]] = {}

    for dataset, truth_map in dataset_truths.items():
        scores = per_dataset_scores.get(dataset)
        if not scores:
            continue

        model_names = list(scores.keys())
        if len(model_names) < 2:
            continue

        msci_values = np.array(
            [scores[name]["msci"] for name in model_names], dtype=float
        )
        cmi_values = np.array(
            [scores[name]["cmi_lb_mean"] for name in model_names], dtype=float
        )

        msci_std = msci_values.std()
        cmi_std = cmi_values.std()
        msci_norm = (msci_values - msci_values.mean()) / (msci_std if msci_std else 1.0)
        cmi_norm = (cmi_values - cmi_values.mean()) / (cmi_std if cmi_std else 1.0)

        overlap = [name for name in model_names if name in truth_map]
        if len(overlap) < 2:
            continue

        indices = np.array([model_names.index(name) for name in overlap], dtype=int)
        truth = np.array([truth_map[name] for name in overlap], dtype=float)

        prepared[dataset] = {
            "msci_norm": msci_norm,
            "cmi_norm": cmi_norm,
            "indices": indices,
            "truth": truth,
            "models": model_names,
        }

    if not prepared:
        return None

    best_objective = float("-inf")
    best_weights: Tuple[float, float] = (0.0, 0.0)
    best_dataset_taus: Dict[str, float] = {}

    def _evaluate_weights(weight_pair: Tuple[float, float]) -> None:
        nonlocal best_objective, best_weights, best_dataset_taus

        aggregate = 0.0
        total_weight = 0.0
        per_dataset_tau: Dict[str, float] = {}

        for dataset, payload in prepared.items():
            combined_all = weight_pair[0] * payload["msci_norm"] + weight_pair[1] * payload[
                "cmi_norm"
            ]
            combined_overlap = combined_all[payload["indices"]]
            tau, _ = weightedtau(combined_overlap, payload["truth"])
            if math.isnan(tau):
                continue
            per_dataset_tau[dataset] = float(tau)
            weight = float(len(payload["truth"]))
            aggregate += weight * float(tau)
            total_weight += weight

        if not per_dataset_tau:
            return

        objective = aggregate / total_weight if total_weight else float(
            np.mean(list(per_dataset_tau.values()))
        )

        if objective > best_objective:
            best_objective = objective
            best_weights = (float(weight_pair[0]), float(weight_pair[1]))
            best_dataset_taus = per_dataset_tau

    for w_msci in search_space:
        for w_cmi in search_space:
            if abs(w_msci) < 1e-12 and abs(w_cmi) < 1e-12:
                continue
            _evaluate_weights((float(w_msci), float(w_cmi)))

    if best_objective == float("-inf"):
        return None

    combined_scores: Dict[str, Dict[str, float]] = {}
    for dataset, payload in prepared.items():
        combined_all = (
            best_weights[0] * payload["msci_norm"] + best_weights[1] * payload["cmi_norm"]
        )
        combined_scores[dataset] = {
            model: float(score)
            for model, score in zip(payload["models"], combined_all, strict=False)
        }

    return best_weights, best_dataset_taus, combined_scores


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    aggregated_results: Dict[str, Dict[str, object]] = {}
    json_payload: Dict[str, Dict[str, object]] = {}
    collected_scores: Dict[str, Dict[str, Dict[str, float]]] = {}

    requested_datasets = _normalise_dataset_argument(args.dataset)
    multi_dataset = len(requested_datasets) > 1

    entries_by_dataset: Dict[str, list[tuple[str, str]]] = {}

    if args.features:
        dataset_for_features: Optional[str] = None
        for candidate in requested_datasets:
            if candidate in DEFAULT_DATASET_MODEL_PATHS:
                dataset_for_features = candidate
                break
        if dataset_for_features is None and DEFAULT_DATASET_MODEL_PATHS:
            dataset_for_features = next(iter(DEFAULT_DATASET_MODEL_PATHS))

        dataset_key = dataset_for_features or "custom"
        dataset_map = DEFAULT_DATASET_MODEL_PATHS.get(dataset_for_features or "", {})
        reverse_lookup = {
            str(Path(path).expanduser().resolve()): model_name
            for model_name, path in dataset_map.items()
        }
        feature_entries: list[tuple[str, str]] = []
        for feature_path in args.features:
            resolved = str(Path(feature_path).expanduser().resolve())
            model_name = reverse_lookup.get(resolved, Path(feature_path).stem)
            feature_entries.append((model_name, feature_path))
        entries_by_dataset[dataset_key] = feature_entries
    else:
        for dataset in requested_datasets:
            dataset_map = DEFAULT_DATASET_MODEL_PATHS.get(dataset)
            if not dataset_map:
                continue
            entries_by_dataset[dataset] = list(dataset_map.items())

        if not entries_by_dataset and DEFAULT_DATASET_MODEL_PATHS:
            dataset, dataset_map = next(iter(DEFAULT_DATASET_MODEL_PATHS.items()))
            entries_by_dataset[dataset] = list(dataset_map.items())

    if not entries_by_dataset:
        raise ValueError(
            "No feature paths provided and no default dataset entries available for the requested dataset(s)."
        )

    multi_dataset_features = len(entries_by_dataset) > 1

    for dataset_name, entries in entries_by_dataset.items():
        if multi_dataset_features:
            print(f"\n### Dataset: {dataset_name}")
        for model_name, feature_path in entries:
            print(
                f"\n=== Evaluating {model_name} ({dataset_name}): {feature_path} ==="
            )
        (
            image_embeddings,
            text_embeddings,
            metadata,
            stored_magnifications,
        ) = load_multimodal_embeddings(feature_path, device)

        _validate_alignment(metadata, image_embeddings, text_embeddings)

        if args.magnifications:
            magnifications = list(args.magnifications)
        elif stored_magnifications:
            magnifications = [mag for mag in stored_magnifications if mag in image_embeddings]
            if not magnifications:
                magnifications = list(sorted(image_embeddings.keys()))
        else:
            magnifications = list(sorted(image_embeddings.keys()))

        msci_result = compute_msci(image_embeddings, text_embeddings, magnifications)
        cmi_results = compute_cmi_lb_across_magnifications(
            image_embeddings,
            text_embeddings,
            magnifications,
            args.temperature,
            min_temperature=args.min_temperature,
            max_temperature=args.max_temperature,
            temperature_steps=args.temperature_steps,
        )

        cmi_avg = float(sum(r.cmi_lb for r in cmi_results.values()) / len(cmi_results))

        collected_scores.setdefault(dataset_name, {})[model_name] = {
            "msci": float(msci_result.msci),
            "cmi_lb_mean": cmi_avg,
            "cmi_lb_per_mag": {
                mag: float(cmi_results[mag].cmi_lb) for mag in magnifications
            },
        }

        print("--- MSCI ---")
        print(f"MSCI score: {msci_result.msci:.6f} (normalised by max variance {msci_result.max_variance:.6f})")
        print(f"Mean variance: {msci_result.mean_variance:.6f}")
        print("Per-region variance (first 10 values):")
        preview = msci_result.per_region_variance[:10].cpu().numpy()
        print(preview)

        print("\n--- CMI-LB ---")
        for mag in magnifications:
            result = cmi_results[mag]
            print(
                f"Magnification {mag}x -> CMI-LB: {result.cmi_lb:.6f} (Lx={result.loss_x:.6f}, Ly={result.loss_y:.6f})"
            )
        print(f"Average CMI-LB across magnifications: {cmi_avg:.6f}")

        aggregated_results.setdefault(dataset_name, {})[feature_path] = {
            "msci": msci_result,
            "cmi_lb": cmi_results,
            "cmi_lb_mean": cmi_avg,
        }

        if args.json:
            json_payload.setdefault(dataset_name, {})[feature_path] = _summarise_for_json(
                msci_result, cmi_results, cmi_avg
            )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(json_payload, handle, indent=2)

    ground_truth = _load_ground_truth(args.ground_truth)
    search_space = np.linspace(args.weight_min, args.weight_max, args.weight_steps)

    joint_optimisation = None
    combined_label = "combined"

    if collected_scores:
        dataset_truths = {
            dataset: gt
            for dataset in requested_datasets
            if (gt := ground_truth.get(dataset))
            if isinstance(gt, Mapping) and gt
        }

        joint_optimisation = _optimise_combined_scores(
            collected_scores,
            dataset_truths,
            search_space,
        )

        if joint_optimisation is not None:
            weights, _, combined_scores = joint_optimisation
            descriptor = (
                "Optimal global combined weights"
                if multi_dataset
                else "Optimal combined weights"
            )
            print(
                f"{descriptor} (w_msci={weights[0]:.3f}, w_cmi={weights[1]:.3f})"
            )
            combined_label = "combined_global" if multi_dataset else "combined"
            for dataset_name, dataset_scores in combined_scores.items():
                prefix = (
                    f"Combined weighted scores for {dataset_name}:"
                    if multi_dataset
                    else "Combined weighted scores:"
                )
                print(prefix)
                for name, value in dataset_scores.items():
                    collected_scores.setdefault(dataset_name, {}).setdefault(name, {})[
                        combined_label
                    ] = value
                    print(f"  {name}: {value:.6f}")

        for dataset in requested_datasets:
            gt = ground_truth.get(dataset)
            if not gt:
                warnings.warn(
                    f"No ground-truth accuracies found for dataset '{dataset}'. Skipping benchmarking.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            dataset_scores = collected_scores.get(dataset)
            if not dataset_scores:
                warnings.warn(
                    f"No computed scores available for dataset '{dataset}'. Skipping benchmarking.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            msci_tau = _compute_weighted_kendall_tau(
                {name: scores["msci"] for name, scores in dataset_scores.items()}, gt
            )
            if msci_tau is not None:
                print(
                    f"Kendall tau_w (MSCI vs ground truth) for {dataset}: {msci_tau:.6f}"
                )
            cmi_tau = _compute_weighted_kendall_tau(
                {name: scores["cmi_lb_mean"] for name, scores in dataset_scores.items()}, gt
            )
            if cmi_tau is not None:
                print(
                    f"Kendall tau_w (CMI-LB mean vs ground truth) for {dataset}: {cmi_tau:.6f}"
                )

            if joint_optimisation is not None:
                weights, dataset_taus, _ = joint_optimisation
                tau_value = dataset_taus.get(dataset)
                if tau_value is not None:
                    descriptor = (
                        "global combined weights"
                        if multi_dataset
                        else "combined weights"
                    )
                    print(
                        f"Kendall tau_w ({descriptor} vs ground truth) for {dataset}: {tau_value:.6f}"
                    )
                else:
                    print(
                        "Unable to derive combined MSCI/CMI-LB weights for benchmarking; insufficient ground-truth overlap."
                    )
            else:
                print(
                    "Unable to derive combined MSCI/CMI-LB weights for benchmarking; insufficient ground-truth overlap."
                )

            # Evaluate per-magnification CMI-LB correlations
            per_mag_scores: Dict[int, Dict[str, float]] = {}
            for name, scores in dataset_scores.items():
                per_mag = scores.get("cmi_lb_per_mag", {})
                if not isinstance(per_mag, Mapping):
                    continue
                for mag, value in per_mag.items():
                    per_mag_scores.setdefault(int(mag), {})[name] = float(value)

            for mag in sorted(per_mag_scores):
                tau_mag = _compute_weighted_kendall_tau(per_mag_scores[mag], gt)
                if tau_mag is None:
                    continue
                print(
                    f"Kendall tau_w (CMI-LB at {mag}x vs ground truth) for {dataset}: {tau_mag:.6f}"
                )

    return aggregated_results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute multimodal transferability metrics.")
    parser.add_argument(
        "features",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of multimodal feature .pth files to evaluate. "
            "If omitted, built-in dataset paths are used when available."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation (e.g. 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--magnifications",
        type=int,
        nargs="*",
        default=None,
        help="Subset of magnifications to evaluate (defaults to all present).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Fixed softmax temperature for CMI-LB. Leave unset to automatically search for an optimal value.",
    )
    parser.add_argument(
        "--min-temperature",
        type=float,
        default=1e-3,
        help="Lower bound of the temperature search interval when --temperature is unset.",
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=1.0,
        help="Upper bound of the temperature search interval when --temperature is unset.",
    )
    parser.add_argument(
        "--temperature-steps",
        type=int,
        default=50,
        help="Number of log-spaced evaluation points used during automatic temperature search.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to store the metrics as JSON.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Optional JSON file containing ground-truth accuracies for Kendall tau benchmarking.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Dataset key(s) used to select ground-truth accuracies. Provide multiple names "
            "separated by spaces or commas to benchmark against several datasets (defaults to all available)."
        ),
    )
    parser.add_argument(
        "--weight-min",
        type=float,
        default=-1.0,
        help="Minimum MSCI weight value when searching the shared combination.",
    )
    parser.add_argument(
        "--weight-max",
        type=float,
        default=1.0,
        help="Maximum MSCI weight value when searching the shared combination.",
    )
    parser.add_argument(
        "--weight-steps",
        type=int,
        default=41,
        help="Number of grid points to evaluate when searching the shared weight.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, object]:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
