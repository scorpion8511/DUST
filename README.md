# DUST
transferability estimation

## Feature Extraction

`extract_features.py` consumes a CSV manifest of multi-scale patches,
splits the rows into named subsets, and writes each model's embeddings to
`MODEL_<split>_features.pth` files. When the manifest already contains a
split column (e.g., `train`/`val`/`test`), pass the column name with
`--split-column` and specify which values map to the train/eval splits via
`--train-splits`/`--eval-splits`. If the column is missing, the script
automatically assigns random splits (80/20 train/eval by default) using a
deterministic seed so repeated runs remain reproducible. Custom ratios can
be provided with repeated `--split-ratio value=fraction` arguments, and the
seed can be overridden via `--random-seed`.

```bash
python extract_features.py manifest.csv out_dir --image-root /path/to/patches --device cuda:0
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
computes an energy metric alongside a histogram-of-intersection (HoI)
separation score and—when the necessary metadata is supplied—the
magnification-scale consistency index (MSCI). A `--device` flag selects the
compute device and an optional `--benchmark` run reports CPU vs. GPU timing:

```bash
python scoring_pipeline.py --device cuda:0 --benchmark
```

All intermediate computations occur on the chosen device; results are
normalized per dataset and combined using learned weights over the energy,
HoI, and MSCI metrics (falling back to the first two when MSCI is
unavailable) before reporting per-dataset Kendall τ correlations with
ground-truth accuracy.

The Gabor Energy metric measures how much diagnostically useful texture a
representation retains by first reshaping embeddings into square “token images,”
filtering them with an oriented, band-pass Gabor bank, and taking the magnitude
of the complex responses. These magnitudes are z-scored, optionally reduced with
PCA, and passed through a linear discriminant analysis (LDA) head fitted on the
training split; the final energy score is the log-sum-exp of the resulting LDA
logits on the evaluation split. This aligns the metric with the code in
`gabor_eng.py`, where the meaningful signal comes from the discriminative power
of the Gabor responses rather than simply pooling their squared amplitudes.

### MSCI for single-modality embeddings

In addition to the Gabor/Fisher metrics, the scoring pipeline can now
compute the magnification-scale consistency index (MSCI) for unimodal
embeddings. Provide a CSV manifest aligned with the evaluation features
that includes a region identifier and magnification per patch, then add
an `msci` block to each model configuration:

```python
dataset_model_paths = {
    "LC": {
        "uni": {
            "train": "/path/to/uni_train_features.pth",
            "eval": "/path/to/uni_eval_features.pth",
            "msci": {
                "manifest": "/path/to/manifest.csv",
                "region_column": "region_id",
                "magnification_column": "patch_scale",
                "magnifications": [5, 10, 20, 40],
            },
        }
    }
}
```

The loader averages embeddings per magnification within each region,
normalises them, and compares the resulting vectors against the region
centroid. Reported statistics include the MSCI score itself, the raw
mean variance across regions, the first ten per-region variances, and
coverage counts so you can verify how many regions satisfied the
requested magnification set. This keeps the unimodal evaluation
comparable to the multimodal MSCI metric while preserving the original
Gabor/Fisher outputs.

### Standalone label-free MSCI utility

For quick experiments where only multi-magnification image patches are
available, `label_free_msci.py` wraps the same MSCI implementation in a
lightweight CLI. The script accepts feature archives produced by either
`extract_features.py` (single-modality) or the multimodal extractor and
computes MSCI without requiring class labels:

```bash
python label_free_msci.py \
    /path/to/features.pth \
    --manifest /path/to/manifest.csv \
    --region-column patch_id \
    --magnification-column patch_scale \
    --magnifications 5 10 20 40
```

If the feature archive already stores `region_ids` and `magnifications` arrays
no manifest is required.  You can also provide `--train-features` when a
separate training split is available; otherwise the evaluation file is reused
for both arguments.  The command prints the MSCI score, mean variance, and the
number of regions contributing to the metric, and an optional `--json` flag
emits a serialised summary for downstream analysis.

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

### CONCH feature extractor

`conch_feature_extraction.py` provides the analogous workflow for CONCH
checkpoints distributed with the open-source CONCH repository. The helper wraps
`conch.open_clip_custom.create_model_from_pretrained`, applies the repository's
pre-processing transform to each cropped patch, and exposes the same CSV schema
and metadata as the TorchScript/PLIP/MUSK pipelines.

```bash
python conch_feature_extraction.py manifest.csv conch_features.pth \
    --hf-token hf_YOUR_WRITE_TOKEN \
    --checkpoint ./checkpoints/CONCH/pytorch_model.bin \
    --model-cfg conch_ViT-B-16 \
    --image-root /home/jovyan/work/tran_est/multires_VL/output \
    --image-column patch_path --text-column generated_text \
    --magnification-column patch_scale --region-column patch_id \
    --strip-region-suffix --metadata-json conch_metadata.json
```

The script instantiates the tokenizer via `conch.open_clip_custom.get_tokenizer`
and reuses the official `tokenize` helper, ensuring captions are processed
exactly as in CONCH's reference snippets. Embeddings are L2-normalised by
default; pass `--no-normalize` to retain raw outputs. The exported metadata
records the resolved region ids, unique captions per region, original CSV
identifiers, and the model/checkpoint parameters used for provenance.

Pass `--hf-token` when the checkpoint resides in a gated Hugging Face
repository; the script performs `huggingface_hub.login` for you. Omit the flag
after authenticating with `huggingface-cli login` or when loading local
checkpoints.

### BiomedCLIP feature extractor

`biomed_feature_extraction.py` targets Microsoft's BiomedCLIP checkpoint via
`open_clip.create_model_from_pretrained`. The script consumes the same CSV
schema as the other multimodal helpers, applies the returned preprocessing
transform to each patch, tokenises captions with the matching BiomedCLIP
tokenizer (respecting the 256-token context window by default), and exports a
`.pth` archive compatible with `multimodal_scoring.py`.

```bash
python biomed_feature_extraction.py manifest.csv biomed_features.pth \
    --image-root /home/jovyan/work/tran_est/multires_VL/output \
    --image-column patch_path --text-column generated_text \
    --magnification-column patch_scale --region-column patch_id \
    --strip-region-suffix --metadata-json biomed_metadata.json
```

L2 normalisation is enabled by default; add `--no-normalize` to keep the raw
encoder outputs. The helper records region ids, deduplicated captions, optional
slide identifiers, and any provided labels in the metadata block so downstream
scoring routines have the same provenance guarantees as the PLIP/CONCH/MUSK
pipelines. Authenticate with `--hf-token` when accessing gated checkpoints on
the Hugging Face Hub (the script will call `huggingface_hub.login`).

### PathGen feature extractor

`pathgen_feature_extraction.py` mirrors the same CSV schema for PathGen CLIP
checkpoints distributed alongside the PathGen release. The script relies on
`open_clip.create_model_and_transforms` to load the requested backbone, applies
the returned preprocessing pipeline to each image patch, and tokenises captions
with the matching `open_clip` tokenizer before exporting a `.pth` archive that
`multimodal_scoring.py` understands out of the box.

```bash
python pathgen_feature_extraction.py manifest.csv pathgen_features.pth \
    --pretrained path/pathgen-clip.pt \
    --model ViT-B-16 --device cuda:0 --precision fp16 \
    --image-root /home/jovyan/work/tran_est/multires_VL/output \
    --image-column patch_path --text-column generated_text \
    --magnification-column patch_scale --region-column patch_id \
    --strip-region-suffix --metadata-json pathgen_metadata.json
```

Embeddings are L2-normalised by default so they can be compared with cosine
similarity; pass `--no-normalize` to retain the raw outputs. Provide
`--hf-token` when the checkpoint lives behind Hugging Face access controls—the
script invokes `huggingface_hub.login` before downloading weights. Metadata in
the exported archive mirrors the other extractors, capturing region ids, unique
captions, per-magnification patch geometry, and provenance for the model
configuration and checkpoint path.

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
sample with the requested encoder (PLIP, MUSK, CONCH, or PathGen), concatenates
the resulting embeddings, and trains a linear classifier.

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
Face. `--model conch` and `--model pathgen` activate the corresponding
`open_clip`-based encoders; supply `--conch-checkpoint` or `--pathgen-pretrained`
to locate the weights (local path or Hugging Face identifier) and reuse
`--hf-token` whenever access control is in place. Validation and test accuracies
are printed at the end of the run.

Embeddings are L2-normalised by default, and repeated captions for the same
region are averaged while retaining every unique description inside the exported
metadata. Additional flags mirror the other extractors: `--filter-column`,
`--strip-region-suffix`, `--drop-missing`, and magnification controls behave the
same way, ensuring MUSK features slot directly into the MSCI and CMI-LB scoring
pipelines.

## Multimodal transferability metrics

`multimodal_scoring.py` implements the Magnification-Scale Consistency
Index (MSCI) and a cross-modal mutual information lower bound (CMI-LB)
for paired image/text embeddings stored in `.pth` files. CMI-LB is the
same symmetric InfoNCE objective used elsewhere in the codebase: for
each magnification, logits = `image_embeddings @ text_embeddingsᵀ / τ`
feed two log-softmax terms (image→text and text→image), and the mean of
`log(N) – CE` across the two directions yields the lower bound on
mutual information. Because each magnification is scored independently
then averaged, the value directly reflects cross-scale semantic
alignment between the vision and text encoders. Each feature file must
contain a `text_embeddings` tensor and an `image_embeddings` dictionary
mapping magnification levels (e.g. 5/10/20) to the corresponding image
feature tensors. To evaluate one or more feature sets, run:

```bash
python multimodal_scoring.py /path/to/plip_features.pth /path/to/musk_features.pth \
    --device cuda:0 --json multimodal_scores.json
```

The scorer automatically searches for the temperature that maximises
the symmetric InfoNCE objective unless a fixed value is supplied via
`--temperature`. The search range can be customised with
`--min-temperature`, `--max-temperature`, and `--temperature-steps`.

For MSCI, the script now reports both the raw mean variance across
magnifications and a normalised score scaled by the theoretical maximum
variance for similarities in ``[-1, 1]``. This keeps the score within
``[0, 1]`` while making it easier to interpret differences across
models when the raw variances are very small.

By default all magnifications present in each feature file are used.
Setting `--magnifications` allows evaluation on a subset (shared across
all inputs), while the `--json` flag stores the resulting metrics in a
single JSON document keyed by feature path for further analysis.


## Classical transferability baselines

`transferability_baselines.py` wraps several classical transferability metrics 
(GBC, SFDA, TransRate, EMMS, NCTI, and H-score) so they can be applied directly 
to the `.pth` artifacts emitted by the extraction pipelines.

```bash
# single-modality embeddings
python transferability_baselines.py single features.pth --manifest manifest.csv --label-column subtype

# multimodal CLIP-style embeddings (image + text)
python transferability_baselines.py multi multimodal_features.pth --manifest manifest.csv --label-column subtype
```

Use `--metrics` to select a subset of scores, `--json` to export results, and
`--emms-backend`/`--emms-model` to choose the language backbone when computing
EMMS.
