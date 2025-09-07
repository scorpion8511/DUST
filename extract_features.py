"""Extract embeddings from multiple pretrained models for a dataset."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm

# Mapping of model names to constructor and checkpoint path.
MODEL_ZOO: Dict[str, Tuple] = {
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


def load_model(name: str, device: torch.device) -> torch.nn.Module:
    """Instantiate a model from ``MODEL_ZOO`` and strip its classification head."""
    ctor, checkpoint = MODEL_ZOO[name]
    model = ctor(weights=None)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    # remove classification head so forward() returns embeddings
    if hasattr(model, "head"):
        model.head = torch.nn.Identity()
    elif hasattr(model, "heads") and hasattr(model.heads, "head"):
        model.heads.head = torch.nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()

    model.to(device)
    model.eval()
    return model


def get_dataloaders(
    data_dir: Path, batch_size: int, split_ratio: float
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Return train and evaluation loaders from a dataset directory.

    Parameters
    ----------
    data_dir: Path
        Root directory containing class subfolders with images.
    batch_size: int
        Number of images per mini-batch.
    split_ratio: float
        Proportion of samples to use for the training split.  The remainder
        constitutes the evaluation split.
    """
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    train_len = int(len(dataset) * split_ratio)
    eval_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(42)
    train_set, eval_set = random_split(dataset, [train_len, eval_len], generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader, dataset.classes


def extract_embeddings(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Run a model over a loader and collect embeddings and labels."""
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="extract", leave=False):
            inputs = inputs.to(device)
            if hasattr(model, "forward_features"):
                outputs = model.forward_features(inputs)
            else:
                outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            features.append(outputs.cpu())
            labels.append(targets)
    embeddings = torch.cat(features)
    labels = torch.cat(labels)
    return {"embeddings": embeddings, "labels": labels}


def save_features(output: Dict[str, torch.Tensor], out_path: Path) -> None:
    """Persist a dictionary of features to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)


def main(
    models: List[str],
    data_dir: Path,
    out_dir: Path,
    device: str,
    batch_size: int,
    split_ratio: float,
) -> None:
    device_obj = torch.device(device)
    train_loader, eval_loader, _ = get_dataloaders(data_dir, batch_size, split_ratio)
    for name in models:
        print(f"Extracting features for {name} on {device}")
        model = load_model(name, device_obj)
        train_feats = extract_embeddings(model, train_loader, device_obj)
        train_out = out_dir / f"{name}_train_features.pth"
        save_features(train_feats, train_out)
        print(f"Saved train features to {train_out}")
        eval_feats = extract_embeddings(model, eval_loader, device_obj)
        eval_out = out_dir / f"{name}_eval_features.pth"
        save_features(eval_feats, eval_out)
        print(f"Saved eval features to {eval_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pretrained features")
    parser.add_argument("data_dir", type=Path, help="Path to dataset root")
    parser.add_argument("out_dir", type=Path, help="Directory to save features")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--split", type=float, default=0.8, help="Train split ratio (default 0.8)"
    )
    parser.add_argument(
        "--models", nargs="*", default=list(MODEL_ZOO.keys()), help="Models to process"
    )
    args = parser.parse_args()
    main(
        args.models,
        args.data_dir,
        args.out_dir,
        args.device,
        args.batch_size,
        args.split,
    )
