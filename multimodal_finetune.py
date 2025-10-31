"""Train a simple classifier on PLIP, MUSK, CONCH, PathGen, or Biomed embeddings.

This script reads an image/text CSV manifest with class labels, encodes each
pair using PLIP, MUSK, CONCH, PathGen, or Biomed encoders, and then trains a
linear classifier on the concatenated embeddings. It mirrors the CSV schema
accepted by the feature extraction utilities (image paths, text prompts,
optional filtering) so existing manifests can be reused for quick accuracy
baselines.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Sample:
    image_path: str
    text: str
    label: int
    index: int
    feature_index: Optional[int] = None
    identifier: Optional[str] = None
    csv_image: Optional[str] = None


@dataclass
class PrecomputedFeatureSet:
    features: torch.Tensor
    labels: Optional[Sequence[object]]
    metadata: Optional[Mapping[str, object]]
    row_indices: Optional[Sequence[int]]


def _coerce_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _sorted_items(mapping: Mapping[object, object]) -> List[Tuple[object, object]]:
    return sorted(mapping.items(), key=lambda item: str(item[0]))


def _gather_image_embeddings(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        tensors: List[torch.Tensor] = []
        for _, tensor in _sorted_items(value):
            tensors.append(_coerce_tensor(tensor))
        if not tensors:
            raise ValueError("image_embeddings mapping did not contain any tensors")
        return torch.cat(tensors, dim=-1)
    raise TypeError("Unsupported image_embeddings format in precomputed features")


def _coerce_label_sequence(value: object) -> Sequence[object]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _extract_labels(payload: Mapping[str, object]) -> Optional[Sequence[object]]:
    if "labels" in payload:
        return _coerce_label_sequence(payload["labels"])
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and "labels" in metadata:
        return _coerce_label_sequence(metadata["labels"])
    return None


def _strip_magnification_suffixes(identifier: str) -> str:
    parts = identifier.split("_")
    if len(parts) <= 1:
        return identifier

    base_parts: List[str] = []
    for part in parts:
        token = part.strip().lower()
        if token.endswith("x"):
            magnitude = token[:-1]
            if magnitude.replace(".", "", 1).isdigit():
                continue
        base_parts.append(part)

    if not base_parts:
        return identifier
    return "_".join(base_parts)


def _maybe_split_token(candidate: str, separator: str) -> List[str]:
    if separator not in candidate:
        return []
    parts = [part for part in candidate.split(separator) if part]
    collapsed = "".join(parts)
    variants = parts[:]
    if collapsed:
        variants.append(collapsed)
    joined = "_".join(parts)
    if joined:
        variants.append(joined)
    return variants


def _normalise_identifier_tokens(
    value: Optional[object], *, allow_partial: bool = True
) -> List[str]:
    if value is None:
        return []

    text = str(value)
    if not text:
        return []

    seen: Set[str] = set()
    tokens: List[str] = []
    queue: List[str] = [text]

    while queue:
        candidate = queue.pop(0)
        if not candidate or candidate in seen:
            continue

        seen.add(candidate)
        tokens.append(candidate)

        normalised = os.path.normpath(candidate)
        if normalised not in seen:
            queue.append(normalised)

        replaced = candidate.replace("\\", "/")
        if replaced not in seen:
            queue.append(replaced)

        normalised_replaced = normalised.replace("\\", "/")
        if normalised_replaced not in seen:
            queue.append(normalised_replaced)

        if any(sep in candidate for sep in ("/", "\\")):
            base = os.path.basename(candidate)
            if base and base not in seen:
                queue.append(base)

        if "::" in candidate:
            colon_squashed = candidate.replace("::", "_")
            if colon_squashed and colon_squashed not in seen:
                queue.append(colon_squashed)
            colon_slash = candidate.replace("::", "/")
            if colon_slash and colon_slash not in seen:
                queue.append(colon_slash)

        root, ext = os.path.splitext(candidate)
        if ext and root and root not in seen:
            queue.append(root)

        stripped = _strip_magnification_suffixes(root if ext and root else candidate)
        if stripped and stripped not in seen:
            queue.append(stripped)

        lowered = candidate.lower()
        if lowered not in seen:
            queue.append(lowered)

        if allow_partial:
            for separator in ("::", "|", "-", ":"):
                for variant in _maybe_split_token(candidate, separator):
                    if variant not in seen:
                        queue.append(variant)

    return tokens


def _build_sample_token_index(samples: Sequence[Sample]) -> Dict[str, List[int]]:
    token_map: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        sample_tokens: Set[str] = set()
        if sample.identifier is not None:
            for token in _normalise_identifier_tokens(sample.identifier, allow_partial=False):
                sample_tokens.add(token)
        for value in (sample.csv_image, sample.image_path):
            for token in _normalise_identifier_tokens(value):
                sample_tokens.add(token)
        for token in sample_tokens:
            token_map.setdefault(token, []).append(idx)
    return token_map


def _select_consistent_candidates(
    candidate_indices: Iterable[int], samples: Sequence[Sample]
) -> Tuple[List[int], Optional[int]]:
    """Return sample indices sharing a single label, otherwise an empty list.

    The alignment metadata for some cached feature sets only records a partial
    identifier (e.g. just the patch identifier without the slide component).
    When we fall back to token based matching we may encounter several manifest
    rows that share the same token but correspond to different slides or even
    different labels.  Instead of propagating the ambiguity, we only accept
    matches where every candidate sample agrees on the label.  The caller can
    then decide whether to reuse the group or keep searching for a less
    ambiguous token.
    """

    indices: List[int] = []
    label_value: Optional[int] = None
    for idx in candidate_indices:
        sample_label = samples[idx].label
        if label_value is None:
            label_value = sample_label
        elif sample_label != label_value:
            return [], None
        indices.append(idx)
    return indices, label_value


def assign_precomputed_indices(samples: Sequence[Sample], features: PrecomputedFeatureSet) -> None:
    if features.features.shape[0] == len(samples):
        return

    assigned = {idx for idx, sample in enumerate(samples) if sample.feature_index is not None}

    if features.row_indices and len(features.row_indices) == features.features.shape[0]:
        for feature_idx, row_idx in enumerate(features.row_indices):
            try:
                row_int = int(row_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= row_int < len(samples):
                sample = samples[row_int]
                if sample.feature_index is None:
                    sample.feature_index = feature_idx
                    assigned.add(row_int)
        if len(assigned) == len(samples):
            return

    metadata = features.metadata or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "Precomputed features contain fewer rows than the manifest and do not provide metadata for alignment."
        )

    csv_mapping = metadata.get("csv_region_mapping")
    region_ids = metadata.get("region_ids")
    if not isinstance(csv_mapping, Mapping) or not isinstance(region_ids, Sequence):
        raise ValueError(
            "Precomputed features contain fewer rows than the manifest and are missing 'csv_region_mapping' metadata."
        )

    token_index = _build_sample_token_index(samples)
    identifier_index: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        if sample.identifier is None:
            continue
        for token in _normalise_identifier_tokens(sample.identifier, allow_partial=False):
            identifier_index.setdefault(token, []).append(idx)
    unmatched_samples = {idx for idx in range(len(samples)) if samples[idx].feature_index is None}

    feature_label_constraints: Dict[int, int] = {}

    for feature_idx, region_id in enumerate(region_ids):
        if not unmatched_samples:
            break
        raw_ids = csv_mapping.get(region_id)
        if raw_ids is None:
            raw_ids = [region_id]
        elif isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        matched = False
        for raw_id in raw_ids:
            identifier_tokens = _normalise_identifier_tokens(raw_id, allow_partial=False)
            for token in identifier_tokens:
                if feature_idx in feature_label_constraints:
                    expected_label = feature_label_constraints[feature_idx]
                else:
                    expected_label = None
                raw_candidates = [
                    idx
                    for idx in identifier_index.get(token, [])
                    if idx in unmatched_samples
                ]
                candidates, label_value = _select_consistent_candidates(
                    raw_candidates, samples
                )
                if not candidates:
                    continue
                if expected_label is not None and label_value is not None and label_value != expected_label:
                    continue
                for sample_idx in candidates:
                    samples[sample_idx].feature_index = feature_idx
                    unmatched_samples.remove(sample_idx)
                if label_value is not None:
                    feature_label_constraints[feature_idx] = label_value
                matched = True
            if matched:
                break
            for token in _normalise_identifier_tokens(raw_id):
                if feature_idx in feature_label_constraints:
                    expected_label = feature_label_constraints[feature_idx]
                else:
                    expected_label = None
                raw_candidates = [
                    idx for idx in token_index.get(token, []) if idx in unmatched_samples
                ]
                if not raw_candidates:
                    continue
                # Require every candidate reached via a fuzzy token to share the
                # same manifest identifier.  This prevents short tokens such as
                # "patch_1" from aligning samples that originate from different
                # slides.
                identifiers = {
                    samples[idx].identifier for idx in raw_candidates if samples[idx].identifier
                }
                if len(identifiers) > 1:
                    continue
                candidates, label_value = _select_consistent_candidates(
                    raw_candidates, samples
                )
                if not candidates:
                    continue
                if expected_label is not None and label_value is not None and label_value != expected_label:
                    continue
                for sample_idx in candidates:
                    samples[sample_idx].feature_index = feature_idx
                    unmatched_samples.remove(sample_idx)
                if label_value is not None:
                    feature_label_constraints[feature_idx] = label_value
                matched = True
        if matched:
            continue

    if unmatched_samples:
        examples: List[str] = []
        for sample_idx in sorted(unmatched_samples)[:5]:
            sample = samples[sample_idx]
            candidate = sample.identifier or sample.csv_image or os.path.basename(sample.image_path)
            examples.append(str(candidate))
        message = (
            "Failed to align the manifest rows with the cached features. "
            "Ensure --id-column matches the identifiers used during feature extraction or regenerate the cache."
        )
        if examples:
            message += f" Example unmatched entries: {', '.join(examples)}."
        raise ValueError(message)


def load_precomputed_feature_set(path: str, normalize: bool) -> PrecomputedFeatureSet:
    payload = torch.load(path, map_location="cpu")
    features_tensor: Optional[torch.Tensor] = None
    labels: Optional[Sequence[object]] = None
    metadata: Optional[Mapping[str, object]] = None
    row_indices: Optional[Sequence[int]] = None

    if isinstance(payload, torch.Tensor):
        features_tensor = payload
    elif isinstance(payload, Mapping):
        metadata_obj = payload.get("metadata")
        if isinstance(metadata_obj, Mapping):
            metadata = metadata_obj
        features_value: Optional[object] = None
        if "features" in payload:
            features_value = payload["features"]
        elif "embeddings" in payload:
            features_value = payload["embeddings"]
        elif "image_embeddings" in payload and "text_embeddings" in payload:
            image_embeddings = _gather_image_embeddings(payload["image_embeddings"])
            text_embeddings = _coerce_tensor(payload["text_embeddings"])
            if normalize:
                image_embeddings = F.normalize(image_embeddings, dim=-1)
                text_embeddings = F.normalize(text_embeddings, dim=-1)
            features_tensor = torch.cat(
                [image_embeddings.float(), text_embeddings.float()], dim=-1
            )
            normalize = False
        else:
            raise ValueError(
                "Precomputed feature file must contain 'features', 'embeddings', or both 'image_embeddings' and 'text_embeddings'."
            )

        if features_tensor is None and features_value is not None:
            features_tensor = _coerce_tensor(features_value).float()

        labels = _extract_labels(payload)

        if "row_indices" in payload:
            row_tensor = _coerce_tensor(payload["row_indices"]).long()
            row_indices = row_tensor.tolist()
    else:
        raise TypeError("Unsupported precomputed feature format")

    if features_tensor is None:
        raise ValueError("Failed to materialise feature tensor from precomputed file")

    if normalize:
        features_tensor = F.normalize(features_tensor, dim=-1)

    return PrecomputedFeatureSet(
        features=features_tensor,
        labels=labels,
        metadata=metadata,
        row_indices=row_indices,
    )


def _maybe_login(token: Optional[str]) -> None:
    """Authenticate with Hugging Face if a token is provided."""

    if not token:
        return

    try:  # pragma: no cover - side effect only used at runtime
        from huggingface_hub import login

        login(token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover - provide context to callers
        raise RuntimeError("Failed to authenticate with Hugging Face.") from exc


def _combine_identifier(row: Mapping[str, str], columns: Sequence[str]) -> Optional[str]:
    parts: List[str] = []
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        token = str(value).strip()
        if token:
            parts.append(token)
    if not parts:
        return None
    return "::".join(parts)


def read_manifest(
    csv_path: str,
    image_root: Optional[str],
    image_column: str,
    text_column: str,
    label_column: str,
    id_column: Optional[str],
    filter_column: Optional[str],
    filter_values: Optional[Sequence[str]],
    *,
    verify_images: bool = True,
    composite_id_columns: Optional[Sequence[str]] = None,
) -> Tuple[List[Sample], Dict[int, str]]:
    """Parse the manifest CSV and return typed samples plus the label map.

    When ``verify_images`` is ``False`` the image paths are not validated on disk,
    which is useful when operating purely on precomputed features.
    """

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
        if id_column and composite_id_columns:
            raise ValueError("--id-column and --id-columns cannot be used together")

        if id_column is not None and id_column not in reader.fieldnames:
            raise ValueError(f"Unknown id column: {id_column}")

        if composite_id_columns:
            missing = [column for column in composite_id_columns if column not in reader.fieldnames]
            if missing:
                raise ValueError(
                    "Unknown id columns for composite identifier: "
                    + ", ".join(missing)
                )

        auto_composite: Optional[Tuple[str, str]] = None
        if not id_column and not composite_id_columns:
            candidate_pairs: Sequence[Tuple[str, str]] = (
                ("slide", "patch"),
                ("slide_id", "patch_id"),
                ("slide", "patch_id"),
                ("slide_id", "patch"),
            )
            for first, second in candidate_pairs:
                if first in reader.fieldnames and second in reader.fieldnames:
                    auto_composite = (first, second)
                    break
            if auto_composite:
                print(
                    "No --id-column supplied; combining columns "
                    f"{auto_composite[0]!r} and {auto_composite[1]!r} to build unique identifiers."
                )

        for row in reader:
            if filter_column and filter_values and row.get(filter_column) not in filter_values:
                continue
            image_value = row[image_column]
            image_path = image_value
            if image_root:
                image_path = os.path.join(image_root, image_path)
            if verify_images and not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing image: {image_path}")
            text = row[text_column]
            raw_label = row[label_column]
            if raw_label is None:
                raise ValueError(
                    f"Missing label value in column {label_column!r} for row {len(samples)}"
                )
            label_name = str(raw_label).strip()
            if not label_name:
                raise ValueError(
                    f"Empty label value in column {label_column!r} for row {len(samples)}"
                )
            if label_name not in label_to_index:
                label_to_index[label_name] = len(label_to_index)
            label_idx = label_to_index[label_name]
            identifier: Optional[str] = None
            if composite_id_columns:
                identifier = _combine_identifier(row, composite_id_columns)
            elif id_column:
                identifier = row[id_column].strip()
            elif auto_composite:
                identifier = _combine_identifier(row, auto_composite)
            if identifier == "":
                identifier = None
            samples.append(
                Sample(
                    image_path=image_path,
                    text=text,
                    label=label_idx,
                    index=len(samples),
                    identifier=identifier,
                    csv_image=image_value,
                )
            )

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


def split_samples_by_feature(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Split samples while keeping cached feature vectors in a single subset."""

    feature_groups: Dict[int, List[Sample]] = {}
    for sample in samples:
        if sample.feature_index is None:
            raise ValueError(
                "split_samples_by_feature requires every sample to define feature_index"
            )
        feature_groups.setdefault(sample.feature_index, []).append(sample)

    for feature_idx, group in feature_groups.items():
        labels = {member.label for member in group}
        if len(labels) > 1:
            raise ValueError(
                "Samples aligned to cached feature index "
                f"{feature_idx} span multiple labels; check the manifest/feature mapping."
            )

    representatives = [group[0] for group in feature_groups.values()]
    train_reps, val_reps, test_reps = split_samples(representatives, train_ratio, val_ratio, seed)

    def expand(reps: Sequence[Sample]) -> List[Sample]:
        expanded: List[Sample] = []
        for rep in reps:
            assert rep.feature_index is not None
            expanded.extend(feature_groups[rep.feature_index])
        return expanded

    return expand(train_reps), expand(val_reps), expand(test_reps)


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

        _maybe_login(hf_token)

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


class CONCHEncoder:
    """Helper that wraps the CONCH open_clip encoder for feature extraction."""

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        device: str,
        precision: str,
        hf_token: Optional[str],
        force_img_size: Optional[int],
    ) -> None:
        _maybe_login(hf_token)

        from PIL import Image  # noqa: F401  (ensure Pillow is available at runtime)
        from conch.open_clip_custom import (
            create_model_from_pretrained,
            get_tokenizer,
            tokenize,
        )

        extra_kwargs = {}
        if force_img_size is not None:
            extra_kwargs["force_img_size"] = int(force_img_size)

        self.model, self.preprocess = create_model_from_pretrained(
            model_cfg,
            checkpoint,
            **extra_kwargs,
        )
        self.tokenizer = get_tokenizer()
        self.tokenize_fn = tokenize

        self.device = torch.device(device)
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.model.to(self.device, dtype=self.dtype).eval()

    def encode(self, samples: Sequence[Sample], batch_size: int, normalize: bool) -> torch.Tensor:
        from PIL import Image

        features: List[torch.Tensor] = []
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            images = [self.preprocess(Image.open(s.image_path).convert("RGB")) for s in chunk]
            texts = [s.text for s in chunk]

            image_tensor = torch.stack(images).to(self.device, dtype=self.dtype)
            text_tokens = self.tokenize_fn(texts=texts, tokenizer=self.tokenizer).to(self.device)

            with torch.inference_mode():
                image_embeddings = self.model.encode_image(image_tensor)
                text_embeddings = self.model.encode_text(text_tokens)

            if normalize:
                image_embeddings = F.normalize(image_embeddings, dim=-1)
                text_embeddings = F.normalize(text_embeddings, dim=-1)

            combined = torch.cat(
                [image_embeddings.float(), text_embeddings.float()], dim=-1
            )
            features.append(combined.cpu())

        return torch.cat(features, dim=0)


class PathGenEncoder:
    """Helper for PathGen checkpoints loaded via open_clip."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str,
        precision: str,
        hf_token: Optional[str],
    ) -> None:
        if not pretrained:
            raise ValueError("--pathgen-pretrained is required when --model pathgen is selected")

        _maybe_login(hf_token)

        from PIL import Image  # noqa: F401  (ensure Pillow is available at runtime)
        import open_clip

        self.device = torch.device(device)
        self.dtype = torch.float16 if precision == "fp16" else torch.float32

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device, dtype=self.dtype).eval()

    def encode(self, samples: Sequence[Sample], batch_size: int, normalize: bool) -> torch.Tensor:
        from PIL import Image

        features: List[torch.Tensor] = []
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            images = [self.preprocess(Image.open(s.image_path).convert("RGB")) for s in chunk]
            texts = [s.text for s in chunk]

            image_tensor = torch.stack(images).to(self.device, dtype=self.dtype)
            text_tokens = self.tokenizer(texts).to(self.device)

            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if self.device.type == "cuda" and self.dtype == torch.float16
                else nullcontext()
            )

            with torch.inference_mode():
                with autocast_ctx:
                    image_embeddings = self.model.encode_image(image_tensor)
                    text_embeddings = self.model.encode_text(text_tokens)

            if normalize:
                image_embeddings = F.normalize(image_embeddings, dim=-1)
                text_embeddings = F.normalize(text_embeddings, dim=-1)

            combined = torch.cat(
                [image_embeddings.float(), text_embeddings.float()], dim=-1
            )
            features.append(combined.cpu())

        return torch.cat(features, dim=0)


class BiomedEncoder:
    """Helper that wraps the BiomedCLIP checkpoint loaded via ``open_clip``."""

    def __init__(
        self,
        model_id: str,
        device: str,
        precision: str,
        context_length: int,
        hf_token: Optional[str],
    ) -> None:
        _maybe_login(hf_token)

        from PIL import Image  # noqa: F401  (ensure Pillow is available at runtime)
        import open_clip

        self.device = torch.device(device)
        self.dtype = torch.float16 if precision == "fp16" else torch.float32

        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_id)
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.context_length = context_length

        self.model.to(self.device, dtype=self.dtype).eval()

    def encode(self, samples: Sequence[Sample], batch_size: int, normalize: bool) -> torch.Tensor:
        from PIL import Image

        features: List[torch.Tensor] = []
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            images = [self.preprocess(Image.open(s.image_path).convert("RGB")) for s in chunk]
            texts = [s.text for s in chunk]

            image_tensor = torch.stack(images).to(self.device, dtype=self.dtype)
            text_tokens = self.tokenizer(texts, context_length=self.context_length).to(self.device)

            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if self.device.type == "cuda" and self.dtype == torch.float16
                else nullcontext()
            )

            with torch.inference_mode():
                with autocast_ctx:
                    image_embeddings = self.model.encode_image(image_tensor)
                    text_embeddings = self.model.encode_text(text_tokens)

            if normalize:
                image_embeddings = F.normalize(image_embeddings, dim=-1)
                text_embeddings = F.normalize(text_embeddings, dim=-1)

            combined = torch.cat(
                [image_embeddings.float(), text_embeddings.float()], dim=-1
            )
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
    conch_checkpoint: Optional[str],
    conch_model_cfg: str,
    conch_precision: str,
    conch_force_img_size: Optional[int],
    pathgen_model: str,
    pathgen_pretrained: Optional[str],
    pathgen_precision: str,
    biomed_model_id: str,
    biomed_precision: str,
    biomed_context_length: int,
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
    if model_name == "conch":
        if not conch_checkpoint:
            raise ValueError("--conch-checkpoint is required when --model conch is selected")
        encoder = CONCHEncoder(
            model_cfg=conch_model_cfg,
            checkpoint=conch_checkpoint,
            device=device,
            precision=conch_precision,
            hf_token=hf_token,
            force_img_size=conch_force_img_size,
        )
        return encoder.encode(samples, batch_size, normalize)
    if model_name == "pathgen":
        encoder = PathGenEncoder(
            model_name=pathgen_model,
            pretrained=pathgen_pretrained or "",
            device=device,
            precision=pathgen_precision,
            hf_token=hf_token,
        )
        return encoder.encode(samples, batch_size, normalize)
    if model_name == "biomed":
        encoder = BiomedEncoder(
            model_id=biomed_model_id,
            device=device,
            precision=biomed_precision,
            context_length=biomed_context_length,
            hf_token=hf_token,
        )
        return encoder.encode(samples, batch_size, normalize)
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
    parser = argparse.ArgumentParser(
        description="Train a classifier using PLIP, MUSK, CONCH, PathGen, or Biomed embeddings"
    )
    parser.add_argument("csv", type=str, help="Path to the image/text manifest")
    parser.add_argument(
        "--model",
        choices=["plip", "musk", "conch", "pathgen", "biomed"],
        required=True,
        help="Encoder to use",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to a .pth file with precomputed features aligned to the manifest order.",
    )
    parser.add_argument("--image-root", type=str, default=None, help="Optional root directory for image paths")
    parser.add_argument("--image-column", type=str, default="patch_path", help="CSV column containing image paths")
    parser.add_argument("--text-column", type=str, default="generated_text", help="CSV column containing text prompts")
    parser.add_argument("--label-column", type=str, default="label", help="CSV column containing class labels")
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help=(
            "Optional CSV column whose values align with identifiers stored in precomputed feature metadata."
        ),
    )
    parser.add_argument(
        "--id-columns",
        nargs="+",
        default=None,
        help=(
            "Optional list of CSV columns whose values are concatenated to form unique identifiers when aligning"
            " precomputed features (for example: --id-columns slide_id patch_id)."
        ),
    )
    parser.add_argument("--filter-column", type=str, default=None, help="Optional column used to filter rows")
    parser.add_argument("--filter-values", nargs="*", default=None, help="Values from --filter-column to keep")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalisation of embeddings before classification")
    parser.add_argument("--musk-checkpoint", type=str, default="hf_hub:xiangjx/musk")
    parser.add_argument("--musk-precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--text-tokenizer", type=str, default=None, help="SentencePiece tokenizer path for MUSK")
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for gated MUSK/CONCH/PathGen checkpoints",
    )
    parser.add_argument("--musk-repo", type=str, default=None, help="Path to a local MUSK clone to add to sys.path")
    parser.add_argument("--text-max-length", type=int, default=100, help="Maximum number of tokens for MUSK captions")
    parser.add_argument(
        "--conch-checkpoint",
        type=str,
        default=None,
        help="Path or identifier for the CONCH checkpoint",
    )
    parser.add_argument(
        "--conch-model-cfg",
        type=str,
        default="conch_ViT-B-16",
        help="Model configuration string for CONCH",
    )
    parser.add_argument(
        "--conch-precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Floating point precision for CONCH inference",
    )
    parser.add_argument(
        "--conch-force-img-size",
        type=int,
        default=None,
        help="Optional forced image size passed to CONCH preprocessing",
    )
    parser.add_argument(
        "--pathgen-model",
        type=str,
        default="ViT-B-16",
        help="Model backbone string passed to open_clip.create_model_and_transforms",
    )
    parser.add_argument(
        "--pathgen-pretrained",
        type=str,
        default=None,
        help="Checkpoint identifier or path for the PathGen model",
    )
    parser.add_argument(
        "--pathgen-precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Floating point precision for PathGen inference",
    )
    parser.add_argument(
        "--biomed-model-id",
        type=str,
        default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        help="Model identifier passed to open_clip.create_model_from_pretrained",
    )
    parser.add_argument(
        "--biomed-precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Floating point precision for Biomed inference",
    )
    parser.add_argument(
        "--biomed-context-length",
        type=int,
        default=256,
        help="Maximum token length supplied to the Biomed tokenizer",
    )
    args = parser.parse_args()

    samples, label_map = read_manifest(
        csv_path=args.csv,
        image_root=args.image_root,
        image_column=args.image_column,
        text_column=args.text_column,
        label_column=args.label_column,
        id_column=args.id_column,
        filter_column=args.filter_column,
        filter_values=args.filter_values,
        verify_images=args.features is None,
        composite_id_columns=args.id_columns,
    )

    precomputed: Optional[PrecomputedFeatureSet] = None
    group_by_feature = False
    if args.features:
        precomputed = load_precomputed_feature_set(
            args.features, normalize=not args.no_normalize
        )
        if precomputed.labels is not None and len(precomputed.labels) != precomputed.features.shape[0]:
            raise ValueError(
                "Precomputed feature labels must align with the stored feature vectors"
            )

        if precomputed.features.shape[0] != len(samples):
            assign_precomputed_indices(samples, precomputed)
            group_by_feature = True
            print(
                "Using precomputed features from "
                f"{args.features} (aligned {len(samples)} manifest rows to {precomputed.features.shape[0]} cached vectors)"
            )
        else:
            print(f"Using precomputed features from {args.features}")

    if group_by_feature:
        train_samples, val_samples, test_samples = split_samples_by_feature(
            samples, args.train_ratio, args.val_ratio, args.seed
        )
    else:
        train_samples, val_samples, test_samples = split_samples(
            samples, args.train_ratio, args.val_ratio, args.seed
        )

    encoder_kwargs = dict(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        normalize=not args.no_normalize,
        musk_checkpoint=args.musk_checkpoint,
        musk_precision=args.musk_precision,
        tokenizer_path=args.text_tokenizer,
        hf_token=args.hf_token,
        musk_repo=args.musk_repo,
        text_max_length=args.text_max_length,
        conch_checkpoint=args.conch_checkpoint,
        conch_model_cfg=args.conch_model_cfg,
        conch_precision=args.conch_precision,
        conch_force_img_size=args.conch_force_img_size,
        pathgen_model=args.pathgen_model,
        pathgen_pretrained=args.pathgen_pretrained,
        pathgen_precision=args.pathgen_precision,
        biomed_model_id=args.biomed_model_id,
        biomed_precision=args.biomed_precision,
        biomed_context_length=args.biomed_context_length,
    )

    def build_features(subset: Sequence[Sample]) -> torch.Tensor:
        if precomputed is not None:
            if not subset:
                return torch.empty(
                    (0, precomputed.features.shape[1]),
                    dtype=precomputed.features.dtype,
                )
            resolved_indices: List[int] = []
            for sample in subset:
                index = sample.feature_index if sample.feature_index is not None else sample.index
                resolved_indices.append(index)
            indices = torch.tensor(resolved_indices, dtype=torch.long)
            return precomputed.features.index_select(0, indices)
        return encode_features(samples=subset, **encoder_kwargs)

    features_train = build_features(train_samples)
    features_val = build_features(val_samples)
    features_test = build_features(test_samples)

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
