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
