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
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_MAGNIFICATIONS: Sequence[int] = (5, 10, 20)


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
) -> tuple[Dict[int, torch.Tensor], torch.Tensor]:
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

    return images, texts


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


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    aggregated_results: Dict[str, object] = {}
    json_payload: Dict[str, object] = {}

    for feature_path in args.features:
        print(f"\n=== Evaluating features: {feature_path} ===")
        image_embeddings, text_embeddings = load_multimodal_embeddings(feature_path, device)

        magnifications = args.magnifications or list(sorted(image_embeddings.keys()))
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

        aggregated_results[feature_path] = {
            "msci": msci_result,
            "cmi_lb": cmi_results,
            "cmi_lb_mean": cmi_avg,
        }

        if args.json:
            json_payload[feature_path] = _summarise_for_json(msci_result, cmi_results, cmi_avg)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(json_payload, handle, indent=2)

    return aggregated_results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute multimodal transferability metrics.")
    parser.add_argument(
        "features",
        type=str,
        nargs="+",
        help="One or more multimodal feature .pth files to evaluate.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> Dict[str, object]:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
