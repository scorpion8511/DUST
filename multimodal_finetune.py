"""Train a simple classifier on PLIP or MUSK embeddings.

This script reads an image/text CSV manifest with class labels, encodes each
pair using either the PLIP or MUSK encoders, and then trains a linear classifier
on the concatenated embeddings. It mirrors the CSV schema accepted by the
feature extraction utilities (image paths, text prompts, optional filtering) so
existing manifests can be reused for quick accuracy baselines.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Sample:
    image_path: str
    text: str
    label: int


def read_manifest(
    csv_path: str,
    image_root: Optional[str],
    image_column: str,
    text_column: str,
    label_column: str,
    filter_column: Optional[str],
    filter_values: Optional[Sequence[str]],
) -> Tuple[List[Sample], Dict[int, str]]:
    """Parse the manifest CSV and return typed samples plus the label map."""

    samples: List[Sample] = []
    label_to_index: Dict[str, int] = {}
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if image_column not in reader.fieldnames or text_column not in reader.fieldnames:
            raise ValueError("CSV must contain both image and text columns")
        if label_column not in reader.fieldnames:
            raise ValueError("CSV must contain a class label column")
        if filter_column is not None and filter_column not in reader.fieldnames:
            raise ValueError(f"Unknown filter column: {filter_column}")

        for row in reader:
            if filter_column and filter_values and row.get(filter_column) not in filter_values:
                continue
            image_path = row[image_column]
            if image_root:
                image_path = os.path.join(image_root, image_path)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing image: {image_path}")
            text = row[text_column]
            label_name = row[label_column]
            if label_name not in label_to_index:
                label_to_index[label_name] = len(label_to_index)
            label_idx = label_to_index[label_name]
            samples.append(Sample(image_path=image_path, text=text, label=label_idx))

    index_to_label = {idx: name for name, idx in label_to_index.items()}
    if not samples:
        raise ValueError("No samples remained after filtering")
    return samples, index_to_label


def split_samples(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Shuffle and split samples into train/val/test subsets."""

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 to leave room for the test split")

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def gather(idxs: Iterable[int]) -> List[Sample]:
        return [samples[i] for i in idxs]

    return gather(train_idx), gather(val_idx), gather(test_idx)


def encode_with_plip(
    samples: Sequence[Sample],
    model_name: str,
    batch_size: int,
    normalize: bool,
    device: str,
) -> torch.Tensor:
    """Encode samples with PLIP and return concatenated image/text features."""

    from PIL import Image
    from plip.plip import PLIP

    encoder = PLIP(model_name)
    features: List[torch.Tensor] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = [Image.open(sample.image_path).convert("RGB") for sample in chunk]
        texts = [sample.text for sample in chunk]
        image_embeddings = torch.as_tensor(encoder.encode_images(images, batch_size=len(images)), dtype=torch.float32)
        text_embeddings = torch.as_tensor(encoder.encode_text(texts, batch_size=len(texts)), dtype=torch.float32)
        if normalize:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        combined = torch.cat([image_embeddings, text_embeddings], dim=-1)
        features.append(combined.cpu())
    return torch.cat(features, dim=0)


class MUSKEncoder:
    """Helper that wraps the MUSK timm backbone and tokenizer."""

    def __init__(
        self,
        model_name: str,
        checkpoint: str,
        device: str,
        precision: str,
        tokenizer_path: str,
        hf_token: Optional[str],
        musk_repo: Optional[str],
        text_max_length: int,
    ) -> None:
        if musk_repo:
            musk_repo = os.path.abspath(musk_repo)
            if musk_repo not in sys.path:
                sys.path.insert(0, musk_repo)
        from musk import utils  # type: ignore
        from musk import modeling as _musk_modeling  # type: ignore  # noqa: F401
        from timm.models import create_model
        import torchvision.transforms as T
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from transformers import XLMRobertaTokenizer

        if hf_token:
            from huggingface_hub import login

            login(hf_token, add_to_git_credential=False)

        self.device = torch.device(device)
        self.precision = torch.float16 if precision == "fp16" else torch.float32
        self.utils = utils
        self.model = create_model(model_name).eval()
        utils.load_model_and_may_interpolate(checkpoint, self.model, "model|module", "")
        self.model.to(self.device, dtype=self.precision).eval()
        self.transform = T.Compose(
            [
                T.Resize(384, interpolation=3, antialias=True),
                T.CenterCrop((384, 384)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ]
        )
        self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
        self.text_max_length = text_max_length

    def encode(self, samples: Sequence[Sample], batch_size: int) -> torch.Tensor:
        from PIL import Image

        features: List[torch.Tensor] = []
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            images = [self.transform(Image.open(s.image_path).convert("RGB")) for s in chunk]
            texts = [s.text for s in chunk]
            image_tensor = torch.stack(images).to(self.device, dtype=self.precision)
            text_ids = []
            pad_masks = []
            for text in texts:
                ids, pad = self.utils.xlm_tokenizer(text, self.tokenizer, max_len=self.text_max_length)
                text_ids.append(torch.tensor(ids, dtype=torch.long))
                pad_masks.append(torch.tensor(pad, dtype=torch.bool))
            text_ids = torch.stack(text_ids).to(self.device)
            pad_masks = torch.stack(pad_masks).to(self.device)
            with torch.inference_mode():
                image_outputs = self.model(
                    image=image_tensor,
                    with_head=True,
                    out_norm=True,
                )[0]
                text_outputs = self.model(
                    text_description=text_ids,
                    padding_mask=pad_masks,
                    with_head=True,
                    out_norm=True,
                )[1]
            combined = torch.cat([image_outputs.float(), text_outputs.float()], dim=-1)
            features.append(combined.cpu())
        return torch.cat(features, dim=0)


def resolve_tokenizer_path(musk_repo: Optional[str]) -> str:
    """Infer the MUSK tokenizer path from a local clone."""

    if musk_repo:
        candidates = [
            os.path.join(musk_repo, "tokenizer.spm"),
            os.path.join(musk_repo, "models", "tokenizer.spm"),
        ]
        parent = os.path.dirname(musk_repo)
        if parent and parent not in ("", musk_repo):
            candidates.extend(
                [
                    os.path.join(parent, "tokenizer.spm"),
                    os.path.join(parent, "models", "tokenizer.spm"),
                ]
            )
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    raise ValueError("Provide --text-tokenizer or point --musk-repo at a MUSK clone with tokenizer.spm")


def encode_features(
    model_name: str,
    samples: Sequence[Sample],
    batch_size: int,
    device: str,
    normalize: bool,
    musk_checkpoint: str,
    musk_precision: str,
    tokenizer_path: Optional[str],
    hf_token: Optional[str],
    musk_repo: Optional[str],
    text_max_length: int,
) -> torch.Tensor:
    if model_name == "plip":
        return encode_with_plip(samples, "vinid/plip", batch_size, normalize, device)
    if model_name == "musk":
        if tokenizer_path is None:
            tokenizer_path = resolve_tokenizer_path(musk_repo)
        encoder = MUSKEncoder(
            model_name="musk_large_patch16_384",
            checkpoint=musk_checkpoint,
            device=device,
            precision=musk_precision,
            tokenizer_path=tokenizer_path,
            hf_token=hf_token,
            musk_repo=musk_repo,
            text_max_length=text_max_length,
        )
        return encoder.encode(samples, batch_size)
    raise ValueError(f"Unsupported model: {model_name}")


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def train_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
) -> nn.Module:
    dataset = FeatureDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device_obj = torch.device(device)
    model = nn.Linear(features.shape[1], num_classes).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for feats, target in loader:
            feats = feats.to(device_obj)
            target = target.to(device_obj)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
    return model


def evaluate(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    device_obj = torch.device(device)
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device_obj))
        preds = logits.argmax(dim=1).cpu()
        correct = (preds == labels).sum().item()
    return correct / len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classifier using PLIP or MUSK embeddings")
    parser.add_argument("csv", type=str, help="Path to the image/text manifest")
    parser.add_argument("--model", choices=["plip", "musk"], required=True, help="Encoder to use")
    parser.add_argument("--image-root", type=str, default=None, help="Optional root directory for image paths")
    parser.add_argument("--image-column", type=str, default="patch_path", help="CSV column containing image paths")
    parser.add_argument("--text-column", type=str, default="generated_text", help="CSV column containing text prompts")
    parser.add_argument("--label-column", type=str, default="label", help="CSV column containing class labels")
    parser.add_argument("--filter-column", type=str, default=None, help="Optional column used to filter rows")
    parser.add_argument("--filter-values", nargs="*", default=None, help="Values from --filter-column to keep")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalisation for PLIP embeddings")
    parser.add_argument("--musk-checkpoint", type=str, default="hf_hub:xiangjx/musk")
    parser.add_argument("--musk-precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--text-tokenizer", type=str, default=None, help="SentencePiece tokenizer path for MUSK")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for MUSK checkpoints")
    parser.add_argument("--musk-repo", type=str, default=None, help="Path to a local MUSK clone to add to sys.path")
    parser.add_argument("--text-max-length", type=int, default=100, help="Maximum number of tokens for MUSK captions")
    args = parser.parse_args()

    samples, label_map = read_manifest(
        csv_path=args.csv,
        image_root=args.image_root,
        image_column=args.image_column,
        text_column=args.text_column,
        label_column=args.label_column,
        filter_column=args.filter_column,
        filter_values=args.filter_values,
    )
    train_samples, val_samples, test_samples = split_samples(
        samples, args.train_ratio, args.val_ratio, args.seed
    )

    features_train = encode_features(
        model_name=args.model,
        samples=train_samples,
        batch_size=args.batch_size,
        device=args.device,
        normalize=not args.no_normalize,
        musk_checkpoint=args.musk_checkpoint,
        musk_precision=args.musk_precision,
        tokenizer_path=args.text_tokenizer,
        hf_token=args.hf_token,
        musk_repo=args.musk_repo,
        text_max_length=args.text_max_length,
    )
    features_val = encode_features(
        model_name=args.model,
        samples=val_samples,
        batch_size=args.batch_size,
        device=args.device,
        normalize=not args.no_normalize,
        musk_checkpoint=args.musk_checkpoint,
        musk_precision=args.musk_precision,
        tokenizer_path=args.text_tokenizer,
        hf_token=args.hf_token,
        musk_repo=args.musk_repo,
        text_max_length=args.text_max_length,
    )
    features_test = encode_features(
        model_name=args.model,
        samples=test_samples,
        batch_size=args.batch_size,
        device=args.device,
        normalize=not args.no_normalize,
        musk_checkpoint=args.musk_checkpoint,
        musk_precision=args.musk_precision,
        tokenizer_path=args.text_tokenizer,
        hf_token=args.hf_token,
        musk_repo=args.musk_repo,
        text_max_length=args.text_max_length,
    )

    labels_train = torch.tensor([sample.label for sample in train_samples], dtype=torch.long)
    labels_val = torch.tensor([sample.label for sample in val_samples], dtype=torch.long)
    labels_test = torch.tensor([sample.label for sample in test_samples], dtype=torch.long)

    classifier = train_classifier(
        features=features_train,
        labels=labels_train,
        num_classes=len(label_map),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )

    val_acc = evaluate(classifier, features_val, labels_val, args.device)
    test_acc = evaluate(classifier, features_test, labels_test, args.device)

    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
