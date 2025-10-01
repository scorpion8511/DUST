# DUST
transferability estimation

## Feature Extraction

`extract_features.py` splits a dataset into training and evaluation
subsets (80/20 by default) and saves each model's embeddings to separate
files:

```bash
python extract_features.py /path/to/images out_dir --device cuda:0
```

This command produces `MODEL_train_features.pth` and
`MODEL_eval_features.pth` for every model in the internal model zoo.

## Fine-tuning from stored features

`finetune.py` can either load raw images or reuse the feature files above.
Provide `--feature_dir` pointing to the directory containing
`MODEL_train_features.pth` and `MODEL_eval_features.pth` to train a
linear classifier on the embeddings:

```bash
python finetune.py --feature_dir out_dir --device cuda:0
```

## Gabor-based scoring

`scoring_pipeline.py` evaluates stored features with Gabor filters and
computes an energy metric alongside a Fisher discriminant score that
measures class separation. A `--device` flag selects the compute device and
an optional `--benchmark` run reports CPU vs. GPU timing:

```bash
python scoring_pipeline.py --device cuda:0 --benchmark
```

All intermediate computations occur on the chosen device; results are
normalized per dataset and combined using learned weights before
reporting per-dataset Kendall τ correlations with ground-truth accuracy.

## Multimodal feature extraction

`multimodal_feature_extraction.py` converts an image-text CSV manifest into the
feature structure consumed by the multimodal scoring utilities. The manifest
should provide `region_id`, `magnification`, `image_path`, `text`, and optional
patch-location columns (`patch_x`, `patch_y`, `patch_size`, `patch_width`,
`patch_height`). Relative image paths can be rooted via `--image-root` and the
loader verifies file existence when `--strict-files` is supplied. For manifests
matching the screenshot you shared (e.g. columns `label`, `is_test`,
`slide_id`, `region_id`, `magnification`, `patch_path`, `prompt_net`,
`patch_x`, `patch_y`, `patch_size`), you can point the script at the relevant
column names without modifying the CSV itself.

The script expects a TorchScript checkpoint with `encode_image` and
`encode_text` entry points. For the locally downloaded encoders you can run:

```bash
python multimodal_feature_extraction.py manifest.csv plip_features.pth \
    --weights /home/jovyan/work/tran_est/MUST/models/plip_model.pth \
    --image-root /home/jovyan/work/tran_est/MUST/patches \
    --image-column patch_path \
    --text-column prompt_net \
    --filter-column is_test --filter-values 0 \
    --device cuda:0 \
    --metadata-json plip_features_metadata.json
```

Repeat the command with the Titan and Conch checkpoints to build comparable
feature sets:

```bash
python multimodal_feature_extraction.py manifest.csv titan_features.pth \
    --weights /home/jovyan/work/tran_est/MUST/models/titan_model.pth \
    --image-column patch_path --text-column prompt_net \
    --filter-column is_test --filter-values 0

python multimodal_feature_extraction.py manifest.csv conch_features.pth \
    --weights /home/jovyan/work/tran_est/MUST/models/conch_model.pth \
    --image-column patch_path --text-column prompt_net \
    --filter-column is_test --filter-values 0
```

If the TorchScript module exposes a custom tokeniser (`tokenize`), raw text can
be passed directly. Otherwise, provide a `--text-token-column` in the CSV that
contains pre-tokenised text as JSON or space-delimited integer ids. Use
`--drop-missing` if you prefer to silently skip regions lacking one or more
requested magnifications instead of terminating with an error.

## Multimodal transferability metrics

`multimodal_scoring.py` implements the Magnification-Scale Consistency
Index (MSCI) and a cross-modal mutual information lower bound (CMI-LB)
for paired image/text embeddings stored in a single `.pth` file. The
file must contain a `text_embeddings` tensor and an
`image_embeddings` dictionary mapping magnification levels (e.g. 5/10/20)
to the corresponding image feature tensors. To evaluate the metrics, run:

```bash
python multimodal_scoring.py /path/to/multimodal_features.pth --device cuda:0 \
    --temperature 0.07 --json multimodal_scores.json
```

By default all magnifications present in the feature file are used.
Setting `--magnifications` allows evaluation on a subset, while the
`--json` flag stores the resulting metrics in JSON format for further
analysis.
