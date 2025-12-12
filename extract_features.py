from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

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
    label_name: str
    region_id: str
    magnification: str
    row_index: int


class MultiScalePatchDataset(Dataset):
    """Dataset that loads multi-scale patches from a manifest."""

    def __init__(
        self,
        samples: Sequence[PatchSample],
        transform: transforms.Compose,
        include_metadata: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.include_metadata = include_metadata

    def __len__(self) -> int:  # noqa: D401 - simple forwarding
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(sample.label_idx, dtype=torch.long)
        if not self.include_metadata:
            return image, label
        region = sample.region_id
        magnification = sample.magnification
        row_index = torch.tensor(sample.row_index, dtype=torch.long)
        return image, label, region, magnification, sample.image_path, row_index


def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
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
    _validate_columns(
        df,
        [image_column, label_column, region_column, magnification_column],
    )
    if filter_column:
        if filter_column not in df.columns:
            raise ValueError(f"Unknown filter column: {filter_column}")
        if filter_values:
            df = df[df[filter_column].isin(filter_values)].copy()
    else:
        df = df.copy()

    def _resolve_path(p: str) -> str:
        """Resolve a manifest path, respecting already-absolute entries.

        The manifest occasionally mixes absolute and relative paths. When an
        image_root is provided we should only prepend it if the manifest entry
        is not already an absolute path that exists. If both the absolute
        manifest entry and the image_root-prefixed version are missing, we fall
        back to the original value so the missing-file report surfaces what the
        user actually supplied.
        """

        if os.path.isabs(p) and os.path.exists(p):
            return p
        if image_root:
            candidate = os.path.join(image_root, p)
            if os.path.exists(candidate):
                return candidate
        return p

    if image_root:
        df[image_column] = df[image_column].map(_resolve_path)

    missing_files = [p for p in df[image_column].tolist() if not os.path.exists(p)]
    if missing_files:
        missing_preview = ", ".join(missing_files[:5])
        raise FileNotFoundError(
            f"{len(missing_files)} images listed in the manifest were not found. "
            f"First missing paths: {missing_preview}"
        )

    return df.reset_index(drop=False).rename(columns={"index": "__row_index"})


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
    mask = df[split_column].isin(values)
    return df[mask].copy()


def assign_random_splits(
    df: pd.DataFrame,
    *,
    split_column: str,
    split_specs: Mapping[str, Sequence[str]],
    ratio_map: Mapping[str, float],
    seed: int,
) -> pd.DataFrame:
    if not split_specs:
        raise ValueError("Cannot generate random splits without split specifications")

    ordered_values: List[str] = []
    for values in split_specs.values():
        for value in values:
            if value not in ordered_values:
                ordered_values.append(value)
    if not ordered_values:
        raise ValueError("Split specifications did not include any split values")

    if ratio_map:
        specified_total = 0.0
        unspecified_values: List[str] = []
        base_ratios: List[float] = []
        for value in ordered_values:
            if value in ratio_map:
                ratio = ratio_map[value]
                if ratio < 0:
                    raise ValueError(
                        f"Split ratio for value '{value}' must be non-negative"
                    )
                base_ratios.append(ratio)
                specified_total += ratio
            else:
                unspecified_values.append(value)
                base_ratios.append(np.nan)
        if specified_total > 1 + 1e-6:
            raise ValueError(
                "Sum of provided split ratios exceeds 1.0; adjust the specifications"
            )
        remaining = max(0.0, 1.0 - specified_total)
        fill_value = (
            remaining / len(unspecified_values)
            if unspecified_values and remaining > 0
            else 0.0
        )
        ratios_array = np.array(
            [fill_value if np.isnan(x) else x for x in base_ratios], dtype=np.float64
        )
        if not np.any(ratios_array):
            ratios_array = np.ones(len(ordered_values), dtype=np.float64)
        else:
            ratios_array = ratios_array / ratios_array.sum()
    else:
        if len(ordered_values) == 1:
            ratios_array = np.ones(1, dtype=np.float64)
        else:
            primary = 0.8
            tail = (1.0 - primary) / (len(ordered_values) - 1)
            ratios_array = np.array(
                [primary] + [tail] * (len(ordered_values) - 1), dtype=np.float64
            )

    total = len(df)
    expected = ratios_array * total
    counts = np.floor(expected).astype(int)
    remainder = total - counts.sum()
    if remainder > 0:
        fractional = expected - counts
        order = np.argsort(-fractional)
        for idx in order[:remainder]:
            counts[idx] += 1

    assignments = np.empty(total, dtype=object)
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(total)
    start = 0
    for value, count in zip(ordered_values, counts):
        end = start + count
        slice_indices = shuffled_indices[start:end]
        assignments[slice_indices] = value
        start = end
    if start < total:
        assignments[shuffled_indices[start:]] = ordered_values[-1]

    df = df.copy()
    df[split_column] = assignments.tolist()
    proportions = {
        value: float((assignments == value).sum()) / float(total)
        for value in ordered_values
    }
    df.attrs["split_proportions"] = proportions
    return df


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
                label_name=label_name,
                region_id=str(row[region_column]),
                magnification=str(row[magnification_column]),
                row_index=int(row["__row_index"]),
            )
        )
    return samples


def load_model(name: str, device: torch.device) -> torch.nn.Module:
    if name not in MODEL_ZOO:
        raise ValueError(f"Unknown model '{name}'. Available options: {sorted(MODEL_ZOO)}")
    ctor, checkpoint = MODEL_ZOO[name]
    model = ctor(weights=None)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    if hasattr(model, "head"):
        model.head = torch.nn.Identity()
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        model.heads.head = torch.nn.Identity()
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    if hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()

    model.to(device)
    model.eval()
    return model


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _extract_batch_features(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, dict):
        if "x" in outputs:
            outputs = outputs["x"]
        else:
            outputs = next(iter(outputs.values()))
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    outputs = outputs.flatten(1)
    return outputs


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    normalize: bool,
) -> Dict[str, torch.Tensor | List[str] | List[int]]:
    embeddings: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    region_ids: List[str] = []
    magnifications: List[str] = []
    paths: List[str] = []
    row_indices: List[int] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="extract", leave=False):
            images, target = batch[0], batch[1]
            metadata = batch[2:] if len(batch) > 2 else None
            images = images.to(device)
            outputs = _extract_batch_features(model, images)
            if normalize:
                outputs = F.normalize(outputs, dim=-1)
            embeddings.append(outputs.cpu())
            labels.append(target.detach().cpu())
            if metadata:
                regions, mags, img_paths, rows = metadata
                region_ids.extend(list(regions))
                magnifications.extend(list(mags))
                paths.extend(list(img_paths))
                row_indices.extend([int(r) for r in rows])

    features = {
        "embeddings": torch.cat(embeddings, dim=0) if embeddings else torch.empty(0),
        "labels": torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long),
    }
    if region_ids:
        features.update(
            {
                "region_ids": region_ids,
                "magnifications": magnifications,
                "paths": paths,
                "row_indices": row_indices,
            }
        )
    return features


def save_manifest_subset(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.drop(columns=["__row_index"], errors="ignore").to_csv(out_path, index=False)


def save_features(out_path: str, payload: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)


def parse_extra_splits(entries: Sequence[str]) -> Dict[str, List[str]]:
    specs: Dict[str, List[str]] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                "Extra split specifications must use the format name=value1,value2"
            )
        name, values = entry.split("=", 1)
        value_list = [v.strip() for v in values.split(",") if v.strip()]
        if not name or not value_list:
            raise ValueError(f"Invalid split specification: {entry}")
        specs[name] = value_list
    return specs


def parse_split_ratios(entries: Sequence[str]) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                "Split ratio specifications must use the format value=fraction"
            )
        name, value = entry.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Invalid split ratio specification: {entry}")
        try:
            fraction = float(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Split ratio specification '{entry}' has a non-numeric fraction"
            ) from exc
        if fraction < 0:
            raise ValueError(f"Split ratio for '{name}' must be non-negative")
        ratios[name] = fraction
    return ratios


def main(args: argparse.Namespace) -> None:
    df = load_manifest(
        args.manifest,
        image_root=args.image_root,
        image_column=args.image_column,
        label_column=args.label_column,
        region_column=args.region_column,
        magnification_column=args.magnification_column,
        filter_column=args.filter_column,
        filter_values=args.filter_values,
    )

    label_map = build_label_map(df, args.label_column)
    label_names = ["" for _ in range(len(label_map))]
    for name, idx in label_map.items():
        label_names[idx] = name

    split_specs: Dict[str, List[str]] = {}
    if args.train_splits:
        split_specs["train"] = args.train_splits
    if args.eval_splits:
        split_specs["eval"] = args.eval_splits
    extra_specs = parse_extra_splits(args.extra_split)
    split_specs.update(extra_specs)

    if not split_specs:
        raise ValueError("At least one split must be specified for feature extraction")

    ratio_map = parse_split_ratios(args.split_ratio)
    if args.split_column and args.split_column not in df.columns:
        df = assign_random_splits(
            df,
            split_column=args.split_column,
            split_specs=split_specs,
            ratio_map=ratio_map,
            seed=args.random_seed,
        )
        print(
            "Split column not found in manifest; generated random assignments using column "
            f"'{args.split_column}'"
        )
        proportions = df.attrs.pop("split_proportions", None)
        if proportions:
            summary = ", ".join(f"{name}: {fraction:.3f}" for name, fraction in proportions.items())
            print(f"Random split proportions -> {summary}")

    subsets: Dict[str, pd.DataFrame] = {}
    for split_name, values in split_specs.items():
        subset = select_split(df, args.split_column, values)
        if subset.empty:
            print(f"Warning: split '{split_name}' with values {values} produced no rows; skipping")
            continue
        subsets[split_name] = subset
        manifest_out = os.path.join(args.out_dir, f"{split_name}_manifest.csv")
        save_manifest_subset(subset, manifest_out)
        print(
            f"Saved filtered manifest for split '{split_name}' with {len(subset)} rows to {manifest_out}"
        )

    if not subsets:
        raise ValueError("No data available after applying split filters")

    device = torch.device(args.device)
    transform = build_transform(args.image_size)
    pin_memory = device.type == "cuda"

    for model_name in args.models:
        print(f"Extracting features for '{model_name}' on device {device}")
        model = load_model(model_name, device)
        for split_name, subset in subsets.items():
            samples = samples_from_dataframe(
                subset,
                image_column=args.image_column,
                label_column=args.label_column,
                region_column=args.region_column,
                magnification_column=args.magnification_column,
                label_map=label_map,
            )
            dataset = MultiScalePatchDataset(samples, transform, include_metadata=True)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            features = extract_embeddings(
                model,
                loader,
                device,
                normalize=args.normalize,
            )
            features.update(
                {
                    "label_names": label_names,
                    "label_mapping": dict(label_map),
                    "split_name": split_name,
                    "split_values": split_specs[split_name],
                    "manifest_path": os.path.abspath(args.manifest),
                    "image_column": args.image_column,
                    "label_column": args.label_column,
                    "region_column": args.region_column,
                    "magnification_column": args.magnification_column,
                    "split_column": args.split_column,
                    "filter_column": args.filter_column,
                    "filter_values": args.filter_values,
                    "image_size": args.image_size,
                    "normalize": args.normalize,
                }
            )
            out_path = os.path.join(args.out_dir, f"{model_name}_{split_name}_features.pth")
            save_features(out_path, features)
            print(
                f"Saved {split_name} features for '{model_name}' with {features['embeddings'].shape[0]} "
                f"samples to {out_path}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pretrained embeddings from multi-scale patches")
    parser.add_argument("manifest", type=str, help="CSV manifest containing patch metadata")
    parser.add_argument("out_dir", type=str, help="Directory to store extracted features")
    parser.add_argument("--models", nargs="*", default=list(MODEL_ZOO.keys()), help="Models to process")
    parser.add_argument("--image-root", type=str, default=None, help="Optional root to prepend to image paths")
    parser.add_argument("--image-column", type=str, default="patch_path", help="CSV column with image paths")
    parser.add_argument("--label-column", type=str, default="subtype", help="CSV column with class labels")
    parser.add_argument("--region-column", type=str, default="patch_id", help="CSV column with region identifiers")
    parser.add_argument("--magnification-column", type=str, default="patch_scale", help="CSV column with magnification levels")
    parser.add_argument("--split-column", type=str, default="split", help="Column that defines dataset splits (train/val/test)")
    parser.add_argument("--train-splits", nargs="*", default=["train"], help="Values that identify training rows in the split column")
    parser.add_argument("--eval-splits", nargs="*", default=["val"], help="Values that identify evaluation rows in the split column")
    parser.add_argument("--extra-split", action="append", default=[], help="Additional split specification of the form name=value1,value2")
    parser.add_argument("--split-ratio", action="append", default=[], help="Optional ratio specification for random splits when the split column is missing (value=fraction)")
    parser.add_argument("--filter-column", type=str, default=None, help="Optional column used to filter rows before processing")
    parser.add_argument("--filter-values", nargs="*", default=None, help="Allowed values for the filter column")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda:0)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224, help="Image resize dimension")
    parser.add_argument("--normalize", action="store_true", help="Apply L2 normalisation to embeddings")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used when generating missing split assignments")
    args = parser.parse_args()
    main(args)
