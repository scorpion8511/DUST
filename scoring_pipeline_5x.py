"""Score TCGA and CAM models with MSCI restricted to the 5× magnification."""

import argparse
from typing import Dict, Any

from gabor_eng import benchmark_runtime

from scoring_pipeline import (
    AVAILABLE_METRICS,
    _validate_metric_selection,
    compute_scores_for_all_datasets,
    normalize_and_combine_scores,
    derive_optimal_weights,
    report_combined_scores,
    compute_kendall_tau_across_datasets,
)


def build_5x_dataset_config() -> Dict[str, Dict[str, Any]]:
    """Return dataset configuration matching :mod:`scoring_pipeline` but 5x-only."""

    msci_shared = {
        "manifest": "/home/jovyan/work/tran_est/multires_txt02.csv",
        "region_column": "patch_id",
        "magnification_column": "patch_scale",
        "label_column": "label",
        "magnifications": [5],
        "allow_single_magnification": True,
    }
    cam_msci_shared = {
        "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
        "region_column": "patch_id",
        "magnification_column": "patch_scale",
        "label_column": "label",
        "magnifications": [5],
        "allow_single_magnification": True,
    }

    return {
        "TCGA": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/uni_eval_features.pth",
                "msci": dict(msci_shared),
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/conch_eval_features.pth",
                "msci": dict(msci_shared),
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/giga_eval_features.pth",
                "msci": dict(msci_shared),
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/phikon_eval_features.pth",
                "msci": dict(msci_shared),
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal/virchow_eval_features.pth",
                "msci": dict(msci_shared),
            },
        },
        "CAM": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/uni_eval_features.pth",
                "msci": dict(cam_msci_shared),
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/conch_eval_features.pth",
                "msci": dict(cam_msci_shared),
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/giga_eval_features.pth",
                "msci": dict(cam_msci_shared),
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/phikon_eval_features.pth",
                "msci": dict(cam_msci_shared),
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/feature_unimodal_cam02/virchow_eval_features.pth",
                "msci": dict(cam_msci_shared),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score models using Gabor features with MSCI restricted to 5x magnification",
    )
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
    args = parser.parse_args()

    dataset_model_paths = build_5x_dataset_config()

    raw_scores = compute_scores_for_all_datasets(dataset_model_paths, device=args.device)
    ground_truth_tcga = {
        "TCGA": {
            "uni": 0.4856,
            "conch": 0.4916,
            "giga": 0.5108,
            "phikon": 0.4675,
            "virchow": 0.4952,
        }
    }
    ground_truth_cam = {
        "CAM": {
            "uni": 0.7656,
            "conch": 0.7903,
            "giga": 0.8567,
            "phikon": 0.8235,
            "virchow": 0.8698,
        }
    }
    ground_truth = {**ground_truth_tcga, **ground_truth_cam}

    selected_metrics = _validate_metric_selection(args.combine_metrics)
    weights, signs = derive_optimal_weights(
        raw_scores, ground_truth, metrics=selected_metrics
    )
    combined_scores = normalize_and_combine_scores(
        raw_scores, metrics=selected_metrics, weights=weights, signs=signs
    )
    report_combined_scores(combined_scores, ground_truth, selected_metrics)
    compute_kendall_tau_across_datasets(combined_scores, ground_truth)

    if args.benchmark:
        print("Benchmark:", benchmark_runtime())


if __name__ == "__main__":
    main()
