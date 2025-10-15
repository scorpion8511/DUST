import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transferability_baselines import kendall_tau_table, mean_tau, weighted_tau


def test_single_dataset_tau_is_one() -> None:
    scores = {
        "toy": {
            "a": {"combined": 0.8},
            "b": {"combined": 0.6},
        }
    }
    ground_truth = {"toy": {"a": 0.9, "b": 0.7}}
    taus = kendall_tau_table(scores, ground_truth, {"combined": 1.0})
    assert "toy" in taus
    assert math.isclose(taus["toy"]["combined"], 1.0)


def test_multi_dataset_tau_and_mean() -> None:
    scores = {
        "toy": {
            "a": {"energy": -0.2, "fisher": 0.8, "combined": 0.75},
            "b": {"energy": -0.5, "fisher": 0.5, "combined": 0.40},
            "c": {"energy": -0.3, "fisher": 0.6, "combined": 0.55},
        },
        "toy_multi": {
            "a": {"energy": -1.0, "fisher": 0.90, "combined": 0.82},
            "b": {"energy": -1.2, "fisher": 0.70, "combined": 0.60},
            "c": {"energy": -1.1, "fisher": 0.85, "combined": 0.78},
        },
    }
    ground_truth = {
        "toy": {"a": 0.92, "b": 0.80, "c": 0.85},
        "toy_multi": {"a": 0.88, "b": 0.75, "c": 0.83},
    }
    metric_signs = {"energy": -1.0, "fisher": 1.0, "combined": 1.0}

    taus = kendall_tau_table(scores, ground_truth, metric_signs)

    toy_energy_oriented = [metric_signs["energy"] * scores["toy"][m]["energy"] for m in ("a", "b", "c")]
    toy_truth = [ground_truth["toy"][m] for m in ("a", "b", "c")]
    expected_toy_energy = weighted_tau(toy_energy_oriented, toy_truth)
    assert pytest.approx(expected_toy_energy) == taus["toy"]["energy"]

    toy_multi_combined = [scores["toy_multi"][m]["combined"] for m in ("a", "b", "c")]
    toy_multi_truth = [ground_truth["toy_multi"][m] for m in ("a", "b", "c")]
    expected_toy_multi_combined = weighted_tau(toy_multi_combined, toy_multi_truth)
    assert pytest.approx(expected_toy_multi_combined) == taus["toy_multi"]["combined"]

    means = mean_tau(taus)
    combined_mean = (taus["toy"]["combined"] + taus["toy_multi"]["combined"]) / 2
    assert pytest.approx(combined_mean) == means["combined"]
