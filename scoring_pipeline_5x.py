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
        "use_labels": False,
        "magnifications": [5],
        "allow_single_magnification": True,
    }
    cam_msci_shared = {
        "manifest": "/home/jovyan/work/tran_est/data_CAM/output/patches.csv",
        "region_column": "patch_id",
        "magnification_column": "patch_scale",
        "use_labels": False,
        "magnifications": [5],
        "allow_single_magnification": True,
    }

    bach_msci = {
        "manifest": "/home/jovyan/work/tran_est/data_BACH/output/patches.csv",
        "region_column": "patch_id",
        "magnification_column": "patch_scale",
        "use_labels": False,
        "magnifications": [5],
        "allow_single_magnification": True,
    }
    bncb_msci = {
        "manifest": "/home/jovyan/work/tran_est/data_BNCB/multiscale_patches/patches.csv",
        "region_column": "patch_id",
        "magnification_column": "patch_scale",
        "use_labels": False,
        "magnifications": [5],
        "allow_single_magnification": True,
    }
    histo_msci = {
        "manifest": "/home/jovyan/work/tran_est/patch_outputs/histo_seg/patches.csv",
        "region_column": "patch_id",
        "magnification_column": "patch_scale",
        "use_labels": False,
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
        "bach": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/uni_eval_features.pth",
                "msci": dict(bach_msci),
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/conch_eval_features.pth",
                "msci": dict(bach_msci),
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/giga_eval_features.pth",
                "msci": dict(bach_msci),
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/phikon_eval_features.pth",
                "msci": dict(bach_msci),
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bach/virchow_eval_features.pth",
                "msci": dict(bach_msci),
            },
        },
        "bncb": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/uni_eval_features.pth",
                "msci": dict(bncb_msci),
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/conch_eval_features.pth",
                "msci": dict(bncb_msci),
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/giga_eval_features.pth",
                "msci": dict(bncb_msci),
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/phikon_eval_features.pth",
                "msci": dict(bncb_msci),
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_bncb/virchow_eval_features.pth",
                "msci": dict(bncb_msci),
            },
        },
        "histo": {
            "uni": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/uni_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/uni_eval_features.pth",
                "msci": dict(histo_msci),
            },
            "conch": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/conch_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/conch_eval_features.pth",
                "msci": dict(histo_msci),
            },
            "giga": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/giga_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/giga_eval_features.pth",
                "msci": dict(histo_msci),
            },
            "phikon": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/phikon_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/phikon_eval_features.pth",
                "msci": dict(histo_msci),
            },
            "virchow": {
                "train": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/virchow_train_features.pth",
                "eval": "/home/jovyan/work/tran_est/MUST/features_unimodal_histo/virchow_eval_features.pth",
                "msci": dict(histo_msci),
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
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.5,
        help="Fraction of embeddings to sample at random for scoring (0 < f <= 1)",
    )
    args = parser.parse_args()

    dataset_model_paths = build_5x_dataset_config()

    raw_scores = compute_scores_for_all_datasets(
        dataset_model_paths,
        device=args.device,
        sample_fraction=args.sample_fraction,
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
    compute_kendall_tau_across_datasets(
        combined_scores, ground_truth, verbose=True
    )

    if args.benchmark:
        print("Benchmark:", benchmark_runtime())


if __name__ == "__main__":
    main()
