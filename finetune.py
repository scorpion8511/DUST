import argparse
import os
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm

MODEL_ZOO = {
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


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    train_ratio: float = 0.75,
    test_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders using ImageFolder labels."""
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_test = int(test_ratio * n_total)
    n_val = n_total - n_train - n_test
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def get_feature_loaders(
    feature_dir: str, model_name: str, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Load precomputed train/eval features for a model."""
    train_path = os.path.join(feature_dir, f"{model_name}_train_features.pth")
    eval_path = os.path.join(feature_dir, f"{model_name}_eval_features.pth")
    train = torch.load(train_path)
    evald = torch.load(eval_path)
    train_ds = TensorDataset(train["embeddings"], train["labels"])
    eval_ds = TensorDataset(evald["embeddings"], evald["labels"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size)
    return train_loader, eval_loader


def create_model(name: str, num_classes: int, device: torch.device) -> nn.Module:
    """Instantiate an architecture and load its pretrained weights."""
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
        else:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "heads"):
        if hasattr(model.heads, "head"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        else:
            in_features = model.heads.in_features
            model.heads = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    model.to(device)
    model.train()
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
    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


def fine_tune(
    model_name: str,
    data_dir: str | None = None,
    feature_dir: str | None = None,
    epochs: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    device = torch.device(device)
    if feature_dir:
        train_loader, eval_loader = get_feature_loaders(feature_dir, model_name, batch_size)
        in_dim = train_loader.dataset.tensors[0].shape[1]
        num_classes = int(train_loader.dataset.tensors[1].max().item()) + 1
        model = nn.Linear(in_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        print(f"\nTraining linear head for {model_name} on {device} for {epochs} epochs")
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate(model, eval_loader, device)
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - val_acc: {val_acc:.4f}"
            )
        accuracy = evaluate(model, eval_loader, device)
        return accuracy
    else:
        train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)
        num_classes = len(train_loader.dataset.dataset.classes)
        model = create_model(model_name, num_classes, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        print(f"\nFine-tuning {model_name} on {device} for {epochs} epochs")
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - val_acc: {val_acc:.4f}"
            )
        accuracy = evaluate(model, test_loader, device)
        return accuracy


def main(
    models_to_run: List[str],
    data_dir: str | None,
    feature_dir: str | None,
    device: str,
):
    if data_dir is None and feature_dir is None:
        raise ValueError("Either data_dir or feature_dir must be provided")
    results: Dict[str, float] = {}
    for name in models_to_run:
        acc = fine_tune(name, data_dir=data_dir, feature_dir=feature_dir, device=device)
        results[name] = acc
        print(f"{name} accuracy: {acc:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune models on LC dataset")
    parser.add_argument("data_dir", type=str, nargs="?", help="Path to LC dataset root")
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Directory containing precomputed *_train_features.pth files",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Training device (cpu or cuda)"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_ZOO.keys()),
        help="Subset of models to fine-tune",
    )
    args = parser.parse_args()
    main(args.models, args.data_dir, args.feature_dir, args.device)
