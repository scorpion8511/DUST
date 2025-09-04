import argparse
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
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


def load_model(name: str, device: torch.device) -> torch.nn.Module:
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


def get_dataloader(data_dir: str, batch_size: int) -> Tuple[DataLoader, List[str]]:
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, dataset.classes


def extract_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
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


def save_features(output: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(output, out_path)


def main(
    models: List[str],
    train_dir: str,
    eval_dir: str,
    out_dir: str,
    device: str,
    batch_size: int,
) -> None:
    device_obj = torch.device(device)
    train_loader, _ = get_dataloader(train_dir, batch_size) if train_dir else (None, None)
    eval_loader, _ = get_dataloader(eval_dir, batch_size) if eval_dir else (None, None)
    for name in models:
        print(f"Extracting features for {name} on {device}")
        model = load_model(name, device_obj)
        if train_loader is not None:
            train_feats = extract_embeddings(model, train_loader, device_obj)
            train_out = os.path.join(out_dir, f"{name}_train_features.pth")
            save_features(train_feats, train_out)
            print(f"Saved train features to {train_out}")
        if eval_loader is not None:
            eval_feats = extract_embeddings(model, eval_loader, device_obj)
            eval_out = os.path.join(out_dir, f"{name}_eval_features.pth")
            save_features(eval_feats, eval_out)
            print(f"Saved eval features to {eval_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pretrained features")
    parser.add_argument("train_dir", type=str, help="Path to training dataset root")
    parser.add_argument("eval_dir", type=str, help="Path to evaluation dataset root")
    parser.add_argument("out_dir", type=str, help="Directory to save features")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--models", nargs="*", default=list(MODEL_ZOO.keys()), help="Models to process"
    )
    args = parser.parse_args()
    main(
        args.models,
        args.train_dir,
        args.eval_dir,
        args.out_dir,
        args.device,
        args.batch_size,
    )
