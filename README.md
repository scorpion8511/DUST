# DUST

A minimal toolkit for dataset transferability estimation.

## MSCI-V Metric

The `msci_v` function computes the **Multi-Scale Cosine Invariance** score for a
set of embeddings that contain multiple magnifications per region.  For each
region the embeddings from different magnifications are normalised and all
pairwise cosine similarities are averaged.  The final MSCI-V score is the mean
across all regions.

```
from msci import msci_v
score = msci_v(embeddings)  # embeddings shape: (regions, magnifications, dim)
```

## Scoring Pipeline

`scoring_pipeline.py` loads multi-magnification features organised as
`dataset/model/*.npy`, computes MSCI-V scores, z-score normalises them **across
all datasets**, and performs a global grid search for metric weights using all
models and datasets together.

```
python scoring_pipeline.py <features_root> [--targets targets.csv]
```

* `features_root`: Root directory containing `dataset/model` folders with `.npy`
  feature files for each magnification.
* `--targets`: Optional CSV with columns `dataset,model,score` used to perform a
  global grid search over metric weights.

The pipeline prints raw and normalised MSCI-V scores.  When a targets file is
provided the optimal global weights and corresponding correlation are reported.

## Preparing Dataset Features

`prepare_dataset.py` converts a BreakHis-style image folder into the
multi-magnification feature layout expected by the scoring pipeline.  The
dataset should be organised as:

```
data_root/
    cancer_type/
        region_id/
            40X/ 100X/ 200X/ 400X/
                *.png   # "X" suffix may be upper- or lower-case
```

Run the preparation script to compute simple colour-histogram embeddings and
write one NumPy file per magnification:

```
python prepare_dataset.py <data_root> <output_root> [--dataset SOB] [--feature-name histogram]
```

This will create files such as
`<output_root>/SOB/histogram/mag40.npy` that can be consumed by
`scoring_pipeline.py`.

## Extracting Features from Pretrained Models

Use `extract_features.py` to generate embeddings from the five provided
pretrained models (`uni`, `conch`, `giga`, `phikon`, `virchow`).  The script
splits the dataset into train/eval subsets, runs each model, and stores the
resulting feature dictionaries as `.pth` files:

```
python extract_features.py <data_root> <out_dir> [--models uni conch giga phikon virchow]
```

Each output file contains an `embeddings` tensor and corresponding `labels`
tensor that can be reused for downstream experiments.
