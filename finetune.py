import argparse
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

MODEL_ZOO = {
    "uni": models.resnet18,
    "conch": models.resnet50,
    "giga": models.densenet121,
    "phikon": models.efficientnet_b0,
    "virchow": models.mobilenet_v3_large,
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


def create_model(name: str, num_classes: int, device: torch.device) -> nn.Module:
    model_fn = MODEL_ZOO[name]
    model = model_fn(weights="DEFAULT")
    model.to(device)
    model.train()
    # Replace classifier with new layer for 5 classes
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
    for inputs, targets in loader:
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
    data_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    device = torch.device(device)
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)
    num_classes = len(train_loader.dataset.dataset.classes)
    model = create_model(model_name, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _ = evaluate(model, val_loader, device)
    accuracy = evaluate(model, test_loader, device)
    return accuracy


def main(models_to_run: List[str], data_dir: str, device: str):
    results: Dict[str, float] = {}
    for name in models_to_run:
        acc = fine_tune(name, data_dir, device=device)
        results[name] = acc
        print(f"{name} accuracy: {acc:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune models on LC dataset")
    parser.add_argument("data_dir", type=str, help="Path to LC dataset root")
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
    main(args.models, args.data_dir, args.device)
