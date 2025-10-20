from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

MODEL_ZOO: Mapping[str, Tuple[object, str]] = {
    "uni": (
        models.vit_l_16,
        "/home/jovyan/work/tran_est/saved_models_and_features_uni_lc01/uni_vit_large_patch16_pretrained_model.pth",
    ),
    "conch": (
        models.vit_b_16,
        "/home/jovyan/work/tran_est/saved_models_and_features_conch_lc01/conch_ViT-B-16_pretrained_model.pth",
    ),
    "giga": (
        models.vit_l_16,
        "/home/jovyan/work/tran_est/saved_models_and_features_giga_lc02/giga_model_vit_large_patch16_224_pretrained_weights.pth",
    ),
    "phikon": (
        models.vit_l_16,
        "/home/jovyan/work/tran_est/saved_models_and_features_phikon_lc01/phikon_v2_pretrained_model.pth",
    ),
    "virchow": (
        models.vit_l_16,
        "/home/jovyan/work/tran_est/saved_models_and_features_vir_lc01/Virchow2_pretrained_model.pth",
    ),
}


@dataclass
class PatchSample:
    image_path: str
    label_idx: int
    region_id: str
    magnification: str


class MultiScalePatchDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[PatchSample],
        transform: transforms.Compose,
        include_metadata: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(sample.label_idx, dtype=torch.long)
        if not self.include_metadata:
            return image, label
        return image, label, sample.region_id, sample.magnification, sample.image_path


def _validate_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col and col not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")


def load_manifest(
    manifest_path: str,
    *,
    image_root: Optional[str],
    image_column: str,
    label_column: str,
    region_column: str,
    magnification_column: str,
    filter_column: Optional[str],
    filter_values: Optional[Sequence[str]],
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    _validate_columns(df, [image_column, label_column, region_column, magnification_column])
    if filter_column:
        if filter_column not in df.columns:
            raise ValueError(f"Unknown filter column: {filter_column}")
        if filter_values:
            df = df[df[filter_column].isin(filter_values)].copy()
    else:
        df = df.copy()
    if image_root:
        df[image_column] = df[image_column].map(lambda p: os.path.join(image_root, p))
    missing = [p for p in df[image_column] if not os.path.exists(p)]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} image files referenced in the manifest are missing. First missing entries: {preview}"
        )
    return df


def build_label_map(df: pd.DataFrame, label_column: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for label in df[label_column].astype(str):
        if label not in mapping:
            mapping[label] = len(mapping)
    return mapping


def select_split(
    df: pd.DataFrame,
    split_column: Optional[str],
    values: Optional[Sequence[str]],
) -> pd.DataFrame:
    if split_column is None or not values:
        return df.copy()
    if split_column not in df.columns:
        raise ValueError(f"Split column '{split_column}' not present in manifest")
    return df[df[split_column].isin(values)].copy()


def samples_from_dataframe(
    df: pd.DataFrame,
    *,
    image_column: str,
    label_column: str,
    region_column: str,
    magnification_column: str,
    label_map: Mapping[str, int],
) -> List[PatchSample]:
    samples: List[PatchSample] = []
    for _, row in df.iterrows():
        label_name = str(row[label_column])
        samples.append(
            PatchSample(
                image_path=str(row[image_column]),
                label_idx=label_map[label_name],
                region_id=str(row[region_column]),
                magnification=str(row[magnification_column]),
            )
        )
    return samples


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def create_model(name: str, num_classes: int, device: torch.device) -> nn.Module:
    if name not in MODEL_ZOO:
        raise ValueError(f"Unknown model '{name}'. Available models: {sorted(MODEL_ZOO)}")
    ctor, checkpoint = MODEL_ZOO[name]
    model = ctor(weights=None)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise TypeError("Unsupported classifier structure for model {name}")
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise TypeError(f"Unable to locate a classification head on model '{name}'")

    model.to(device)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for batch in tqdm(loader, desc="train", leave=False):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
    return running_loss / max(total, 1)


def evaluate(model: nn.Module, loader: Optional[DataLoader], device: torch.device) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else float("nan")


def create_feature_loaders(
    feature_dir: str,
    model_name: str,
    train_split: str,
    eval_split: str,
    test_split: Optional[str],
    batch_size: int,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], int]:
    def _load(split: str) -> Mapping[str, torch.Tensor]:
        path = os.path.join(feature_dir, f"{model_name}_{split}_features.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing features for split '{split}': {path}")
        return torch.load(path)

    train = _load(train_split)
    evald = _load(eval_split)
    test = _load(test_split) if test_split else None

    def _to_tensor_dataset(features: Mapping[str, torch.Tensor]) -> TensorDataset:
        embeddings = features["embeddings"]
        labels = features["labels"]
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings)
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)
        return TensorDataset(embeddings, labels)

    train_loader = DataLoader(
        _to_tensor_dataset(train),
        batch_size=batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        _to_tensor_dataset(evald),
        batch_size=batch_size,
    )
    test_loader = (
        DataLoader(_to_tensor_dataset(test), batch_size=batch_size)
        if test is not None
        else None
    )
    def _max_label(features: Mapping[str, torch.Tensor]) -> int:
        labels = features["labels"]
        if isinstance(labels, torch.Tensor):
            return int(labels.max().item()) if labels.numel() else -1
        array = np.asarray(labels)
        return int(array.max()) if array.size else -1

    num_classes = _max_label(train)
    num_classes = max(num_classes, _max_label(evald))
    if test is not None:
        num_classes = max(num_classes, _max_label(test))
    num_classes += 1
    return train_loader, eval_loader, test_loader, num_classes


def prepare_image_dataloaders(
    df: pd.DataFrame,
    *,
    image_column: str,
    label_column: str,
    region_column: str,
    magnification_column: str,
    label_map: Mapping[str, int],
    split_column: Optional[str],
    train_splits: Sequence[str],
    val_splits: Sequence[str],
    test_splits: Sequence[str],
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    splits = {
        "train": select_split(df, split_column, train_splits),
        "val": select_split(df, split_column, val_splits) if val_splits else pd.DataFrame(columns=df.columns),
        "test": select_split(df, split_column, test_splits) if test_splits else pd.DataFrame(columns=df.columns),
    }

    if splits["train"].empty:
        raise ValueError("Training split produced zero samples; check split configuration")

    datasets_dict: Dict[str, Optional[Dataset]] = {}
    for split_name, subset in splits.items():
        if subset.empty:
            if split_name != "train":
                print(f"Warning: split '{split_name}' produced no samples; it will be skipped")
            datasets_dict[split_name] = None
            continue
        samples = samples_from_dataframe(
            subset,
            image_column=image_column,
            label_column=label_column,
            region_column=region_column,
            magnification_column=magnification_column,
            label_map=label_map,
        )
        include_meta = split_name != "train"
        datasets_dict[split_name] = MultiScalePatchDataset(
            samples,
            transform,
            include_metadata=include_meta,
        )

    train_loader = DataLoader(
        datasets_dict["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            datasets_dict["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if datasets_dict["val"] is not None
        else None
    )
    test_loader = (
        DataLoader(
            datasets_dict["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if datasets_dict["test"] is not None
        else None
    )
    return train_loader, val_loader, test_loader


def fine_tune(
    model_name: str,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    num_classes: int,
) -> Dict[str, float]:
    model = create_model(model_name, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}" +
            (" - val_acc: {:.4f}".format(val_acc) if not pd.isna(val_acc) else "")
        )
    final_val = evaluate(model, val_loader, device)
    final_test = evaluate(model, test_loader, device)
    return {"val_acc": final_val, "test_acc": final_test}


def train_linear_head(
    model_name: str,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    num_classes: int,
) -> Dict[str, float]:
    input_dim = train_loader.dataset.tensors[0].shape[1]
    model = nn.Linear(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = 0.0
        total = 0
        model.train()
        for inputs, targets in tqdm(train_loader, desc="train", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{epochs} - loss: {train_loss / max(total, 1):.4f}" +
            (" - val_acc: {:.4f}".format(val_acc) if not pd.isna(val_acc) else "")
        )
    final_val = evaluate(model, val_loader, device)
    final_test = evaluate(model, test_loader, device)
    return {"val_acc": final_val, "test_acc": final_test}


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    if args.feature_dir:
        print("Using precomputed features for fine-tuning")
    else:
        if not args.manifest:
            raise ValueError("A manifest must be provided when --feature_dir is not used")

    for model_name in args.models:
        print(f"\n=== Fine-tuning {model_name} ===")
        if args.feature_dir:
            train_loader, val_loader, test_loader, num_classes = create_feature_loaders(
                args.feature_dir,
                model_name,
                train_split=args.feature_train_split,
                eval_split=args.feature_eval_split,
                test_split=args.feature_test_split,
                batch_size=args.batch_size,
            )
            results = train_linear_head(
                model_name,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=num_classes,
            )
        else:
            manifest = load_manifest(
                args.manifest,
                image_root=args.image_root,
                image_column=args.image_column,
                label_column=args.label_column,
                region_column=args.region_column,
                magnification_column=args.magnification_column,
                filter_column=args.filter_column,
                filter_values=args.filter_values,
            )
            label_map = build_label_map(manifest, args.label_column)
            transform = build_transform(args.image_size)
            train_loader, val_loader, test_loader = prepare_image_dataloaders(
                manifest,
                image_column=args.image_column,
                label_column=args.label_column,
                region_column=args.region_column,
                magnification_column=args.magnification_column,
                label_map=label_map,
                split_column=args.split_column,
                train_splits=args.train_splits,
                val_splits=args.val_splits,
                test_splits=args.test_splits,
                transform=transform,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            results = fine_tune(
                model_name,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=len(label_map),
            )
        test_acc = results.get("test_acc")
        val_acc = results.get("val_acc")
        if not pd.isna(test_acc):
            print(f"{model_name} test accuracy: {test_acc:.4f}")
        elif not pd.isna(val_acc):
            print(f"{model_name} validation accuracy: {val_acc:.4f}")
        else:
            print("No evaluation split available to report accuracy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune pretrained ViT models on multi-scale patches")
    parser.add_argument("manifest", type=str, nargs="?", help="CSV manifest with patch metadata and subtype labels")
    parser.add_argument("--feature_dir", type=str, default=None, help="Directory containing precomputed *_features.pth files")
    parser.add_argument("--models", nargs="*", default=list(MODEL_ZOO.keys()), help="Models to fine-tune")
    parser.add_argument("--image-root", type=str, default=None, help="Optional root to prepend to image paths")
    parser.add_argument("--image-column", type=str, default="patch_path")
    parser.add_argument("--label-column", type=str, default="subtype")
    parser.add_argument("--region-column", type=str, default="patch_id")
    parser.add_argument("--magnification-column", type=str, default="patch_scale")
    parser.add_argument("--split-column", type=str, default="split")
    parser.add_argument("--train-splits", nargs="*", default=["train"])
    parser.add_argument("--val-splits", nargs="*", default=["val"])
    parser.add_argument("--test-splits", nargs="*", default=["test"])
    parser.add_argument("--filter-column", type=str, default=None)
    parser.add_argument("--filter-values", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--feature-train-split", type=str, default="train")
    parser.add_argument("--feature-eval-split", type=str, default="eval")
    parser.add_argument("--feature-test-split", type=str, default=None)
    args = parser.parse_args()
    main(args)
