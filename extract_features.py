"""Extract multi-magnification embeddings from pretrained models."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm.auto import tqdm

# Magnifications expected in the dataset
MAGNIFICATIONS = ["40", "100", "200", "400"]

# Mapping of model names to constructor and checkpoint path
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


class ImagePathDataset(Dataset):
    """Dataset that loads images from a list of file paths."""

    def __init__(self, paths: List[Path], transform: transforms.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB"))


def load_model(name: str, device: torch.device) -> torch.nn.Module:
    """Instantiate a model from ``MODEL_ZOO`` and strip its classification head."""
    ctor, checkpoint = MODEL_ZOO[name]
    model = ctor(weights=None)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

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


def embed_image_dir(
    model: torch.nn.Module,
    img_dir: Path,
    device: torch.device,
    batch_size: int,
    transform: transforms.Compose,
) -> torch.Tensor:
    """Return the average embedding of all images in ``img_dir``."""
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no images found in {img_dir}")
    dataset = ImagePathDataset(paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats: List[torch.Tensor] = []
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            if hasattr(model, "forward_features"):
                outputs = model.forward_features(inputs)
            else:
                outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            feats.append(outputs.cpu())
    return torch.cat(feats, dim=0).mean(dim=0)


def collect_regions(data_root: Path) -> Tuple[List[Path], List[int]]:
    """Return a list of region directories and their integer labels."""
    regions: List[Path] = []
    labels: List[int] = []
    cancer_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    for label, cancer_dir in enumerate(cancer_dirs):
        for region_dir in sorted(p for p in cancer_dir.iterdir() if p.is_dir()):
            regions.append(region_dir)
            labels.append(label)
    return regions, labels


def find_mag_dir(region: Path, mag: str) -> Path:
    """Locate the directory for a given magnification within ``region``."""
    candidates = [region / mag, region / f"{mag}x", region / f"{mag}X"]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"no images for {region} at {mag}X")


def extract_features(
    models: List[str],
    data_root: Path,
    out_root: Path,
    dataset: str,
    device: str,
    batch_size: int,
) -> None:
    device_obj = torch.device(device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    regions, labels = collect_regions(data_root)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    for name in models:
        print(f"Extracting features for {name} on {device}")
        model = load_model(name, device_obj)
        feats: Dict[str, List[torch.Tensor]] = {mag: [] for mag in MAGNIFICATIONS}
        for region in tqdm(regions, desc=name):
            for mag in MAGNIFICATIONS:
                img_dir = find_mag_dir(region, mag)
                emb = embed_image_dir(model, img_dir, device_obj, batch_size, transform)
                feats[mag].append(emb)
        out_dir = out_root / dataset / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for mag, emb_list in feats.items():
            arr = torch.stack(emb_list, dim=0)
            torch.save({"embeddings": arr, "labels": labels_tensor}, out_dir / f"mag{mag}.pth")
        print(f"Saved features for {name} to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract multi-magnification features")
    parser.add_argument("data_root", type=Path, help="Root of BreakHis-style dataset")
    parser.add_argument("out_root", type=Path, help="Directory to store features")
    parser.add_argument("--dataset", default="SOB", help="Dataset name for output path")
    parser.add_argument("--models", nargs="*", default=list(MODEL_ZOO.keys()), help="Models to process")
    parser.add_argument("--device", default="cpu", help="Device to run models on")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    args = parser.parse_args()

    extract_features(
        args.models,
        args.data_root,
        args.out_root,
        args.dataset,
        args.device,
        args.batch_size,
    )
