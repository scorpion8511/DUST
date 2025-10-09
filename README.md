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
`encode_text` entry points. Checkpoints exported via `torch.package`
are also supported – the loader will fall back to `PackageImporter` when
the TorchScript archive omits `constants.pkl`. For the locally downloaded
encoders you can run:

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
requested magnifications instead of terminating with an error. When your CSV
stores magnification coverage inside the `region_id` (e.g. values like
`patch_0_5x_10x_20x` for a single patch observed at multiple scales), enable
`--strip-region-suffix` so the loader collapses those entries to a shared
identifier (`patch_0`). The original CSV identifiers are preserved inside the
exported metadata under `csv_region_mapping` for traceability.

Magnification entries may either be plain numbers (``5``) or tokens such as
``5x``/``10×``; the loader automatically extracts the numeric component before
grouping crops by magnification.

### PLIP convenience extractor

When you would rather call the native `plip` library instead of TorchScript
encoders, use `plip_feature_extraction.py`. The script consumes the same CSV
schema (including patch metadata, filtering, and suffix stripping) and produces
a `.pth` archive shaped exactly like the TorchScript pipeline, making it
compatible with `multimodal_scoring.py`.

```bash
python plip_feature_extraction.py manifest.csv plip_native_features.pth \
    --image-root /home/jovyan/work/tran_est/multires_VL/output \
    --image-column patch_path --text-column generated_text \
    --magnification-column patch_scale --region-column patch_id \
    --strip-region-suffix --filter-column is_test --filter-values 0 \
    --metadata-json plip_native_metadata.json
```

By default the loader instantiates `PLIP('vinid/plip')` and L2-normalises both
modalities (mirroring the public usage example). Supply `--model` to point at a
different checkpoint or `--no-normalize` to export the raw embeddings. When a
region appears with multiple captions (e.g. separate descriptions for each
magnification) the script averages their embeddings and stores every unique
caption in the exported metadata. The resulting metadata dictionary captures the
resolved region ids, original CSV identifiers, per-magnification patch geometry,
and the PLIP model identifier for auditability.

### MUSK timm extractor

`musk_feature_extraction.py` now follows the official MUSK instructions: the
script boots a timm backbone, loads weights via
`musk.utils.load_model_and_may_interpolate`, and reproduces the resize / crop /
normalisation recipe from the repository demos. The helper accepts the same CSV
schema as the TorchScript and PLIP pipelines while outputting a `.pth` archive
that is immediately compatible with `multimodal_scoring.py`.

```bash
python musk_feature_extraction.py manifest.csv musk_features.pth \
    --hf-token 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 \
    --model-name musk_large_patch16_384 \
    --checkpoint hf_hub:xiangjx/musk \
    --device cuda:0 --precision fp16 \
    --image-root /home/jovyan/work/tran_est/multires_VL/output \
    --image-column patch_path --text-column generated_text \
    --magnification-column patch_scale --region-column patch_id \
    --strip-region-suffix --metadata-json musk_metadata.json
```

Pass `--hf-token` to authenticate against the gated Hugging Face repository (the
script calls `huggingface_hub.login` for you) or leave it unset after running
`huggingface-cli login` manually. When the MUSK code lives outside the Python
path, point `--musk-repo` at a local clone to mirror `sys.path` setup from the
demo notebook. If the clone exposes `models/tokenizer.spm` the extractor wraps
the official XLM-R SentencePiece tokenizer via `musk.utils.xlm_tokenizer`,
matching the snippet from `demo.ipynb`. Provide `--text-tokenizer` to point at a
custom `.spm` file or fall back to an `open_clip` vocabulary; combine it with
`--text-max-length` to truncate captions as needed. When omitted, the extractor
derives the model's text context length (1024 tokens for the released MUSK
checkpoints) so the positional embeddings match the repository implementation.

## Multimodal fine-tuning for classification

`multimodal_finetune.py` offers a lightweight way to benchmark simple
classification accuracy from the paired image/text datasets already used for
feature extraction. Unlike the scoring utilities, which only need image–text
pairs, the fine-tuning script **requires** a class label for every row. Think of
the manifest as a standard supervised dataset where each image and its caption
are paired with a ground-truth class (e.g., diagnosis, tissue type). The CSV
must therefore include an explicit label column in addition to the image and
text fields.

The script accepts the same manifest structure as the extraction utilities
(image paths, text descriptions, optional filtering) along with a class label
column. It splits the manifest into train/validation/test subsets, encodes each
sample with either PLIP or MUSK, concatenates the resulting embeddings, and
trains a linear classifier.

```bash
python multimodal_finetune.py manifest.csv \
    --model plip \
    --image-root /home/jovyan/work/tran_est/multires_VL/output \
    --image-column patch_path --text-column generated_text \
    --label-column diagnosis --filter-column split --filter-values train val test
```

Switch `--model musk` to fine-tune on MUSK embeddings. When using MUSK, provide
`--text-tokenizer` or point `--musk-repo` at a local clone containing
`tokenizer.spm`, and pass `--hf-token` if the checkpoint is gated on Hugging
Face. Validation and test accuracies are printed at the end of the run.

Embeddings are L2-normalised by default, and repeated captions for the same
region are averaged while retaining every unique description inside the exported
metadata. Additional flags mirror the other extractors: `--filter-column`,
`--strip-region-suffix`, `--drop-missing`, and magnification controls behave the
same way, ensuring MUSK features slot directly into the MSCI and CMI-LB scoring
pipelines.

## Multimodal transferability metrics

`multimodal_scoring.py` implements the Magnification-Scale Consistency
Index (MSCI) and a cross-modal mutual information lower bound (CMI-LB)
for paired image/text embeddings stored in `.pth` files. Each feature
file must contain a `text_embeddings` tensor and an
`image_embeddings` dictionary mapping magnification levels (e.g. 5/10/20)
to the corresponding image feature tensors. To evaluate one or more
feature sets, run:

```bash
python multimodal_scoring.py /path/to/plip_features.pth /path/to/musk_features.pth \
    --device cuda:0 --json multimodal_scores.json
```

The scorer automatically searches for the temperature that maximises
the symmetric InfoNCE objective unless a fixed value is supplied via
`--temperature`. The search range can be customised with
`--min-temperature`, `--max-temperature`, and `--temperature-steps`.

By default all magnifications present in each feature file are used.
Setting `--magnifications` allows evaluation on a subset (shared across
all inputs), while the `--json` flag stores the resulting metrics in a
single JSON document keyed by feature path for further analysis.
