"""Extract MUSK vision-language features from CSV manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from multimodal_feature_extraction import ImageTextCSVDataset


class _SentencePieceTokenizerWrapper:
    """Adapter around Hugging Face tokenizers backed by SentencePiece files."""

    def __init__(
        self,
        base_tokenizer,
        musk_utils_module,
        max_length: Optional[int],
    ) -> None:
        self._tokenizer = base_tokenizer
        self._musk_utils = musk_utils_module
        self._max_length = max_length
        # expose the underlying tokenizer so MUSK utilities can reuse it
        self.base_tokenizer = base_tokenizer

        pad_id = getattr(base_tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_token = getattr(base_tokenizer, "pad_token", None)
            if pad_token is not None:
                try:  # pragma: no cover - best effort fallback
                    pad_id = base_tokenizer.convert_tokens_to_ids(pad_token)
                except Exception:
                    pad_id = None
        if pad_id is None:
            pad_id = 0
        self.pad_token_id = int(pad_id)

    def _resolve_max_length(self) -> int:
        if self._max_length and self._max_length > 0:
            return int(self._max_length)

        candidates: List[int] = []
        for attr in ("model_max_length", "max_len_single_sentence", "max_model_input_sizes"):
            value = getattr(self._tokenizer, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(int(value))
            elif isinstance(value, Mapping):
                for v in value.values():
                    if isinstance(v, int) and v > 0:
                        candidates.append(int(v))
        for candidate in candidates:
            if 0 < candidate < 1_000_000:
                return candidate
        return 512

    def __call__(self, texts: Sequence[str]) -> Mapping[str, Tensor]:
        max_len = self._resolve_max_length()
        encoded = self._tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )

        text_tensor = encoded["input_ids"].to(dtype=torch.long)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            padding_mask = attention_mask.to(dtype=torch.bool)
            padding_mask = ~padding_mask
        else:
            padding_mask = text_tensor.eq(self.pad_token_id)

        return {
            "text_description": text_tensor,
            "padding_mask": padding_mask,
            "attention_mask": attention_mask,
        }

    def __getattr__(self, name: str):  # pragma: no cover - passthrough
        return getattr(self._tokenizer, name)

try:  # pragma: no cover - optional dependency
    from huggingface_hub import login as hf_login
except Exception:  # pragma: no cover - handled at runtime
    hf_login = None  # type: ignore


@dataclass
class RegionAccumulator:
    """Stores embeddings and metadata for a spatial region."""

    region_id: str
    text_sum: Optional[Tensor] = None
    text_count: int = 0
    text_values: List[str] = field(default_factory=list)
    images: Dict[int, Tensor] = field(default_factory=dict)
    metadata: Dict[str, object] = field(
        default_factory=lambda: {"patches": {}, "samples": [], "csv_region_ids": set()}
    )


def build_collate_fn():
    """Create a collate function that preserves PIL images and metadata."""

    def collate(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        images = [item["image"] for item in batch]
        texts = [item.get("text") for item in batch]
        if any(text is None for text in texts):
            raise ValueError(
                "Encountered missing text entries; ensure --text-column is present in the CSV."
            )

        magnifications = [int(item["magnification"]) for item in batch]
        region_ids = [str(item["region_id"]) for item in batch]
        metadata = [item["metadata"] for item in batch]

        return {
            "images": images,
            "texts": [text or "" for text in texts],
            "magnifications": magnifications,
            "region_ids": region_ids,
            "metadata": metadata,
        }

    return collate


def _to_device(value: Union[Tensor, Mapping[str, Tensor]], device: torch.device) -> Union[Tensor, Mapping[str, Tensor]]:
    if isinstance(value, Mapping):
        return {key: tensor.to(device) for key, tensor in value.items()}
    return value.to(device)


class MUSKBackbone:
    """Wrapper around the official MUSK timm checkpoint."""

    def __init__(
        self,
        model_name: str,
        checkpoint: str,
        device: str,
        dtype: str = "fp16",
        ms_augment: bool = True,
        image_size: int = 384,
        antialias: bool = True,
        text_tokenizer: Optional[str] = None,
        login_token: Optional[str] = None,
        musk_repo_root: Optional[str] = None,
        strict_checkpoint_key: str = "model|module",
        text_max_length: Optional[int] = None,
    ) -> None:
        repo_root_abs: Optional[str] = None
        if musk_repo_root:
            repo_root_abs = os.path.abspath(musk_repo_root)
            sys.path.insert(0, repo_root_abs)

        self._ensure_fairscale_stub()

        try:
            from musk import modeling as _musk_modeling  # type: ignore  # noqa: F401
            from musk import utils as musk_utils  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError(
                "Could not import the `musk` package. Clone https://github.com/lilab-stanford/MUSK"
                " and provide --musk-repo or install it via `pip install -e .`."
            ) from exc

        self.musk_utils = musk_utils
        
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError("timm is required for MUSK feature extraction; install it via `pip install timm`."
            ) from exc

        try:
            import torchvision.transforms as T
            from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError(
                "torchvision is required for MUSK preprocessing; install it via `pip install torchvision`."
            ) from exc

        if login_token:
            if hf_login is None:  # pragma: no cover - handled at runtime
                raise ImportError(
                    "huggingface_hub is required for authentication; install it via `pip install huggingface-hub`."
                )
            hf_login(token=login_token, add_to_git_credential=False)

        precision_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
        dtype = dtype.lower()
        if dtype not in precision_map:
            raise ValueError(f"Unsupported precision '{dtype}'. Choose from {sorted(precision_map)}.")
        target_dtype = precision_map[dtype]

        self.device = torch.device(device)
        if self.device.type == "cpu" and target_dtype != torch.float32:
            target_dtype = torch.float32

        self.ms_augment = ms_augment
        self.target_dtype = target_dtype
        self.text_max_length = text_max_length
        self.musk_repo_root = repo_root_abs

        self.model = timm.models.create_model(model_name, pretrained=False)
        musk_utils.load_model_and_may_interpolate(
            checkpoint,
            self.model,
            strict_checkpoint_key,
            "",
        )
        self.model.to(device=self.device, dtype=self.target_dtype)
        self.model.eval()

        if self.text_max_length is None:
            inferred_length = self._infer_model_text_length()
            if inferred_length is None:
                inferred_length = 1024
            # Some checkpoints report very small default context windows (for
            # instance 16 or 100) even though the positional embeddings expect
            # much longer sequences (typically 1024).  When that happens the
            # model will crash once the padding mask is broadcast inside the
            # attention blocks.  Guard against those misreports by clamping to
            # the true positional embedding length when available and otherwise
            # falling back to a conservative default of 1024 tokens.
            self.text_max_length = max(int(inferred_length), 1024)

        interpolation_attr = getattr(T, "InterpolationMode", None)
        if interpolation_attr is not None:
            interpolation = getattr(interpolation_attr, "BICUBIC", 3)
        else:
            interpolation = 3
        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=interpolation, antialias=antialias),
                T.CenterCrop((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ]
        )

        tokenizer_hint = text_tokenizer
        if tokenizer_hint is None and self.musk_repo_root is not None:
            default_sentencepiece = os.path.join(self.musk_repo_root, "models", "tokenizer.spm")
            if os.path.exists(default_sentencepiece):
                tokenizer_hint = default_sentencepiece
            else:
                root_tokenizer = os.path.join(self.musk_repo_root, "tokenizer.spm")
                if os.path.exists(root_tokenizer):
                    tokenizer_hint = root_tokenizer

        self.tokenizer, self.tokenizer_name = self._build_tokenizer(tokenizer_hint)

    def _infer_default_tokenizer_name(self) -> Optional[str]:
        cfg = getattr(self.model, "default_cfg", None)
        if isinstance(cfg, Mapping):
            for key in ("text_tokenizer", "tokenizer", "hf_tokenizer"):
                value = cfg.get(key)
                if value:
                    return str(value)
        return None

    def _infer_model_text_length(self) -> Optional[int]:
        candidate_keys = (
            "text_len",
            "text_length",
            "max_text_len",
            "max_text_length",
            "context_length",
        )
        for key in candidate_keys:
            value = getattr(self.model, key, None)
            if isinstance(value, int) and value > 0:
                return value

        beit3 = getattr(self.model, "beit3", None)
        if beit3 is not None:
            encoder = getattr(beit3, "encoder", None)
            embed_positions = getattr(encoder, "embed_positions", None)
            if embed_positions is not None:
                for attr in ("num_embeddings", "weight", "pe", "embeddings"):
                    candidate = getattr(embed_positions, attr, None)
                    if isinstance(candidate, torch.Tensor) and candidate.ndim >= 1:
                        return int(candidate.shape[0])
                    if isinstance(candidate, int) and candidate > 0:
                        return int(candidate)
                if hasattr(embed_positions, "weight"):
                    weight = getattr(embed_positions, "weight")
                    if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
                        return int(weight.shape[0])

        cfg = getattr(self.model, "default_cfg", None)
        if isinstance(cfg, Mapping):
            for key in ("text_len", "text_length", "context_length"):
                value = cfg.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        return None

    def _build_tokenizer(self, override: Optional[str]) -> Tuple[object, Optional[str]]:
        tokenizer_name = override or self._infer_default_tokenizer_name()

        tokeniser: Optional[object] = None
        resolved_name: Optional[str] = tokenizer_name
        errors: List[str] = []

        if tokenizer_name:
            try:
                if hasattr(self.musk_utils, "get_tokenizer"):
                    tokeniser = self.musk_utils.get_tokenizer(tokenizer_name)
                elif hasattr(self.musk_utils, "create_tokenizer"):
                    tokeniser = self.musk_utils.create_tokenizer(tokenizer_name)
            except Exception as exc:  # pragma: no cover - best effort
                errors.append(f"musk.utils tokenizer '{tokenizer_name}' failed: {exc}")

        if tokeniser is None:
            tokeniser, resolved_name, error = self._build_sentencepiece_tokenizer(
                tokenizer_name=override or tokenizer_name,
            )
            if error:
                errors.append(error)

        if tokeniser is None:
            try:
                import open_clip  # type: ignore

                resolved = override or tokenizer_name or "ViT-L-14"
                tokeniser = open_clip.get_tokenizer(resolved)
                resolved_name = resolved
            except Exception as exc:  # pragma: no cover - handled at runtime
                errors.append(f"open_clip.get_tokenizer failed: {exc}")

        if tokeniser is None:
            error_message = "Unable to instantiate a tokenizer for MUSK."
            if errors:
                error_message += " Errors: " + " | ".join(errors)
            raise RuntimeError(error_message)

        return tokeniser, resolved_name

    def _candidate_sentencepiece_paths(self, hint: Optional[str]) -> List[str]:
        candidates: List[str] = []

        def _normalise(path: str) -> Optional[str]:
            expanded = os.path.abspath(os.path.expanduser(path))
            if os.path.isdir(expanded):
                possible = [
                    os.path.join(expanded, "tokenizer.spm"),
                    os.path.join(expanded, "models", "tokenizer.spm"),
                ]
                return next((p for p in possible if os.path.exists(p)), None)
            if os.path.isfile(expanded):
                return expanded
            return None

        if hint:
            normalised = _normalise(hint)
            if normalised:
                candidates.append(normalised)
            elif self.musk_repo_root:
                joined = _normalise(os.path.join(self.musk_repo_root, hint))
                if joined:
                    candidates.append(joined)

        if self.musk_repo_root:
            for suffix in ("tokenizer.spm", os.path.join("models", "tokenizer.spm")):
                candidate = os.path.join(self.musk_repo_root, suffix)
                if os.path.exists(candidate):
                    candidates.append(os.path.abspath(candidate))

        return list(dict.fromkeys(candidates))

    def _build_sentencepiece_tokenizer(
        self,
        tokenizer_name: Optional[str],
    ) -> Tuple[Optional[object], Optional[str], Optional[str]]:
        candidates = self._candidate_sentencepiece_paths(tokenizer_name)
        if not candidates:
            if self.musk_repo_root:
                return (
                    None,
                    tokenizer_name,
                    f"no tokenizer.spm found under {self.musk_repo_root}; pass --text-tokenizer",
                )
            return None, tokenizer_name, None

        try:
            from transformers import XLMRobertaTokenizer
        except Exception as exc:  # pragma: no cover - handled at runtime
            return None, tokenizer_name, f"transformers XLMRobertaTokenizer unavailable: {exc}"

        last_error: Optional[str] = None
        for candidate in candidates:
            try:
                base_tokenizer = XLMRobertaTokenizer(candidate, use_fast=False)
                wrapper = _SentencePieceTokenizerWrapper(
                    base_tokenizer,
                    self.musk_utils,
                    self.text_max_length,
                )
                name = candidate
                if hasattr(base_tokenizer, "name_or_path"):
                    name = str(base_tokenizer.name_or_path)
                return wrapper, name, None
            except Exception as exc:  # pragma: no cover - handled at runtime
                last_error = f"XLMRobertaTokenizer failed for '{candidate}': {exc}"
        return None, tokenizer_name, last_error

    def _ensure_fairscale_stub(self) -> None:
        try:  # pragma: no cover - optional dependency
            import fairscale.nn  # type: ignore  # noqa: F401
            return
        except ImportError:
            pass

        def _identity_wrapper(module, *args, **kwargs):
            return module

        fairscale_module = types.ModuleType("fairscale")
        nn_module = types.ModuleType("fairscale.nn")
        nn_module.checkpoint_wrapper = _identity_wrapper  # type: ignore[attr-defined]
        nn_module.wrap = _identity_wrapper  # type: ignore[attr-defined]
        fairscale_module.nn = nn_module  # type: ignore[attr-defined]

        sys.modules.setdefault("fairscale", fairscale_module)
        sys.modules.setdefault("fairscale.nn", nn_module)

    def _run_model(self, **kwargs):
        common_kwargs = dict(with_head=False, out_norm=False, return_global=True)
        if "image" in kwargs and self.ms_augment:
            common_kwargs["ms_aug"] = True
        try:
            return self.model(**common_kwargs, **kwargs)
        except TypeError:
            return self.model(**kwargs)

    @staticmethod
    def _ensure_2d_tensor(tensor: Tensor) -> Tensor:
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor

    def encode_images(self, images: Sequence[object]) -> Tensor:
        tensors = [self.transform(image).to(self.target_dtype) for image in images]
        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "encode_image"):
                outputs = self.model.encode_image(batch)
            else:
                outputs = self._run_model(image=batch)

        if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
            image_features = outputs[0]
        else:
            image_features = outputs

        return image_features.detach().to("cpu", dtype=torch.float32)

    def _prepare_text_inputs(self, texts: Sequence[str]) -> Mapping[str, Tensor]:
        tokenizer = self.tokenizer
        tokens = tokenizer(texts)

        pad_id = getattr(getattr(tokenizer, "base_tokenizer", tokenizer), "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "pad_token_id", 0)
        pad_id = int(pad_id)

        attention_mask_tensor: Optional[Tensor] = None

        if isinstance(tokens, Mapping):
            mapping: Dict[str, Tensor] = {key: torch.as_tensor(value) for key, value in tokens.items()}
            text_tensor: Optional[Tensor] = None
            for candidate in ("text_description", "input_ids", "text"):
                if candidate in mapping:
                    text_tensor = mapping[candidate]
                    if candidate != "text_description":
                        mapping.pop(candidate)
                    break
            if text_tensor is None:
                raise ValueError("Tokenizer output missing token ids; provide a MUSK-compatible tokenizer.")
            text_tensor = torch.as_tensor(text_tensor, dtype=torch.long)
            text_tensor = self._ensure_2d_tensor(text_tensor)
            if self.text_max_length is not None and self.text_max_length > 0:
                limit = int(self.text_max_length)
                text_tensor = text_tensor[..., :limit]
            padding_mask = text_tensor.eq(pad_id)
            result = {
                "text_description": text_tensor,
                "padding_mask": padding_mask,
            }
            return result

        if isinstance(tokens, (tuple, list)) and len(tokens) >= 1:
            text_tensor = torch.as_tensor(tokens[0], dtype=torch.long)
        else:
            text_tensor = torch.as_tensor(tokens, dtype=torch.long)

        text_tensor = self._ensure_2d_tensor(text_tensor)
        if self.text_max_length is not None and self.text_max_length > 0:
            limit = int(self.text_max_length)
            text_tensor = text_tensor[..., :limit]
        padding_mask = text_tensor.eq(pad_id)
        return {
            "text_description": text_tensor,
            "padding_mask": padding_mask,
        }

    def encode_text(self, texts: Sequence[str]) -> Tensor:
        tokens = self._prepare_text_inputs(texts)
        tokens = _to_device(tokens, self.device)

        with torch.no_grad():
            if hasattr(self.model, "encode_text"):
                if isinstance(tokens, Mapping):
                    text_features = self.model.encode_text(**tokens)
                else:
                    text_features = self.model.encode_text(tokens)
            else:
                kwargs = tokens if isinstance(tokens, Mapping) else {"text": tokens}
                outputs = self._run_model(**kwargs)
                if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                    text_features = outputs[1]
                else:
                    text_features = outputs

        return text_features.detach().to("cpu", dtype=torch.float32)


def extract_musk_embeddings(
    dataloader: DataLoader,
    backbone: MUSKBackbone,
    normalize: bool,
    magnifications: Optional[Sequence[int]],
    drop_missing: bool,
) -> Dict[str, object]:
    regions: "OrderedDict[str, RegionAccumulator]" = OrderedDict()
    observed_magnifications: set[int] = set()

    for batch in dataloader:
        images = batch["images"]
        texts = batch["texts"]

        image_embeddings = backbone.encode_images(images)
        text_embeddings = backbone.encode_text(texts)

        if normalize:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

        for idx, region_id in enumerate(batch["region_ids"]):
            magnification = int(batch["magnifications"][idx])
            observed_magnifications.add(magnification)
            accumulator = regions.setdefault(region_id, RegionAccumulator(region_id=region_id))

            embedding_image = image_embeddings[idx]
            embedding_text = text_embeddings[idx]

            if accumulator.text_sum is None:
                accumulator.text_sum = embedding_text.clone()
            else:
                accumulator.text_sum = accumulator.text_sum + embedding_text
            accumulator.text_count += 1
            accumulator.text_values.append(texts[idx])

            accumulator.images[magnification] = embedding_image
            accumulator.metadata.setdefault("samples", []).append(batch["metadata"][idx])
            patch = batch["metadata"][idx].get("patch")
            if patch is not None:
                accumulator.metadata.setdefault("patches", {})[magnification] = patch
            original_id = batch["metadata"][idx].get("csv_region_id", region_id)
            accumulator.metadata.setdefault("csv_region_ids", set()).add(original_id)

    if magnifications is None:
        ordered_magnifications = sorted(observed_magnifications)
    else:
        ordered_magnifications = [int(m) for m in magnifications]
        missing = set(ordered_magnifications) - observed_magnifications
        if missing:
            raise ValueError(
                f"Requested magnifications {sorted(missing)} were not found in the dataset."
            )

    region_ids: List[str] = []
    text_embeddings_out: List[Tensor] = []
    raw_texts: List[Optional[object]] = []
    csv_region_mapping: Dict[str, List[str]] = {}
    patches = {mag: [] for mag in ordered_magnifications}

    for region_id, accumulator in regions.items():
        if accumulator.text_sum is None or accumulator.text_count == 0:
            if drop_missing:
                continue
            raise ValueError(f"Missing text embedding for region {region_id}.")

        text_mean = accumulator.text_sum / float(accumulator.text_count)
        if normalize:
            text_mean = F.normalize(text_mean.unsqueeze(0), dim=-1).squeeze(0)

        region_ids.append(region_id)
        text_embeddings_out.append(text_mean)

        if accumulator.text_values:
            unique_texts = list(OrderedDict.fromkeys(accumulator.text_values))
            raw_texts.append(unique_texts if len(unique_texts) > 1 else unique_texts[0])
        else:
            raw_texts.append(None)

        csv_ids = accumulator.metadata.get("csv_region_ids", set())
        if not csv_ids:
            csv_ids = {region_id}
        csv_region_mapping[region_id] = sorted(csv_ids)

        for mag in ordered_magnifications:
            embedding = accumulator.images.get(mag)
            if embedding is None:
                if drop_missing:
                    break
                raise ValueError(
                    f"Region {region_id} does not contain an image for magnification {mag}."
                )
            patches[mag].append(accumulator.metadata.get("patches", {}).get(mag))
        else:
            continue

        region_ids.pop()
        text_embeddings_out.pop()
        raw_texts.pop()
        csv_region_mapping.pop(region_id, None)
        for mag in ordered_magnifications:
            if patches[mag]:
                patches[mag].pop()

    if not region_ids:
        raise ValueError(
            "All regions were filtered out – disable --drop-missing or verify magnification coverage."
        )

    text_tensor = torch.stack(text_embeddings_out, dim=0)
    image_tensors: Dict[int, Tensor] = {}
    for mag in ordered_magnifications:
        embeddings = [regions[r].images.get(mag) for r in region_ids]
        if any(embed is None for embed in embeddings):
            missing_regions = [region_ids[idx] for idx, embed in enumerate(embeddings) if embed is None]
            raise ValueError(
                f"The following regions are missing magnification {mag}: {missing_regions}"
            )
        image_tensors[mag] = torch.stack(embeddings, dim=0)  # type: ignore[arg-type]

    metadata = {
        "region_ids": region_ids,
        "texts": raw_texts,
        "patches": patches,
        "csv_region_mapping": csv_region_mapping,
    }

    return {
        "image_embeddings": image_tensors,
        "text_embeddings": text_tensor,
        "metadata": metadata,
        "magnifications": ordered_magnifications,
    }


def save_outputs(features: Mapping[str, object], output_path: str, metadata_json: Optional[str]) -> None:
    torch.save(features, output_path)
    if metadata_json:
        serialisable = {
            "region_ids": features["metadata"]["region_ids"],
            "texts": features["metadata"]["texts"],
            "patches": features["metadata"]["patches"],
            "csv_region_mapping": features["metadata"]["csv_region_mapping"],
            "magnifications": features["magnifications"],
        }
        with open(metadata_json, "w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract MUSK features from a CSV manifest using the official timm checkpoint."
    )
    parser.add_argument("csv", type=str, help="Path to the CSV file with image/text pairs.")
    parser.add_argument("output", type=str, help="Destination .pth file for the extracted features.")

    parser.add_argument(
        "--model-name",
        type=str,
        default="musk_large_patch16_384",
        help="MUSK timm model identifier (default: musk_large_patch16_384).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="hf_hub:xiangjx/musk",
        help="Checkpoint identifier passed to musk.utils.load_model_and_may_interpolate.",
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="model|module",
        help="Regex describing the checkpoint key containing weights (default: model|module).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device used for MUSK inference (default: cuda:0).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Floating point precision used during inference (default: fp16).",
    )
    parser.add_argument(
        "--no-ms-augment",
        dest="ms_augment",
        action="store_false",
        help="Disable multi-scale augmentation when encoding images.",
    )
    parser.set_defaults(ms_augment=True)
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Input resolution for MUSK preprocessing (default: 384).",
    )
    parser.add_argument(
        "--no-antialias",
        dest="antialias",
        action="store_false",
        help="Disable anti-aliased resizing during preprocessing.",
    )
    parser.set_defaults(antialias=True)
    parser.add_argument(
        "--text-tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer name; defaults to the model configuration or ViT-L-14 for open_clip.",
    )
    parser.add_argument(
        "--text-max-length",
        type=int,
        default=None,
        help="Optional maximum number of tokens fed to the MUSK text encoder.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token used to login before downloading checkpoints.",
    )
    parser.add_argument(
        "--musk-repo",
        type=str,
        default=None,
        help="Optional path to a local MUSK repository clone added to PYTHONPATH before importing.",
    )

    parser.add_argument("--image-root", type=str, default=None, help="Optional root to prepend to relative image paths.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of samples per DataLoader batch (default: 16).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes (default: 4).",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable L2 normalisation for image and text embeddings (enabled by default).",
    )
    parser.set_defaults(normalize=True)

    parser.add_argument("--region-column", type=str, default="region_id")
    parser.add_argument("--magnification-column", type=str, default="magnification")
    parser.add_argument("--image-column", type=str, default="image_path")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--text-token-column", type=str, default=None)
    parser.add_argument("--patch-x-column", type=str, default=None)
    parser.add_argument("--patch-y-column", type=str, default=None)
    parser.add_argument("--patch-width-column", type=str, default=None)
    parser.add_argument("--patch-height-column", type=str, default=None)
    parser.add_argument("--patch-size-column", type=str, default=None)
    parser.add_argument(
        "--default-patch-size",
        type=float,
        default=None,
        help="Fallback square size when only patch coordinates are supplied.",
    )
    parser.add_argument(
        "--strict-files",
        action="store_true",
        help="Raise if image files referenced in the CSV are missing.",
    )
    parser.add_argument(
        "--strip-region-suffix",
        action="store_true",
        help="Remove trailing magnification tokens (e.g. 'patch_0_5x') when grouping regions.",
    )
    parser.add_argument("--filter-column", type=str, default=None)
    parser.add_argument(
        "--filter-values",
        type=str,
        nargs="*",
        default=None,
        help="Keep rows whose filter-column value matches one of the provided tokens.",
    )
    parser.add_argument(
        "--magnifications",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of magnifications to retain; defaults to those found in the CSV.",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Discard regions lacking at least one of the requested magnifications instead of failing.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="Optional path to write the metadata dictionary as JSON alongside the .pth archive.",
    )

    return parser


def run(args: argparse.Namespace) -> Dict[str, object]:
    dataset = ImageTextCSVDataset(
        csv_path=args.csv,
        image_root=args.image_root,
        region_column=args.region_column,
        magnification_column=args.magnification_column,
        image_column=args.image_column,
        text_column=args.text_column,
        text_token_column=args.text_token_column,
        location_x_column=args.patch_x_column,
        location_y_column=args.patch_y_column,
        location_width_column=args.patch_width_column,
        location_height_column=args.patch_height_column,
        location_size_column=args.patch_size_column,
        default_patch_size=args.default_patch_size,
        strict_files=args.strict_files,
        strip_region_suffix=args.strip_region_suffix,
        filter_column=args.filter_column,
        filter_values=args.filter_values,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=build_collate_fn(),
    )

    backbone = MUSKBackbone(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.precision,
        ms_augment=args.ms_augment,
        image_size=args.image_size,
        antialias=args.antialias,
        text_tokenizer=args.text_tokenizer,
        login_token=args.hf_token,
        musk_repo_root=args.musk_repo,
        strict_checkpoint_key=args.checkpoint_key,
        text_max_length=args.text_max_length,
    )

    features = extract_musk_embeddings(
        dataloader=loader,
        backbone=backbone,
        normalize=args.normalize,
        magnifications=args.magnifications,
        drop_missing=args.drop_missing,
    )

    metadata = features.get("metadata")
    if isinstance(metadata, dict):
        metadata.update(
            {
                "musk_model": args.model_name,
                "musk_checkpoint": args.checkpoint,
                "precision": args.precision,
                "ms_augment": args.ms_augment,
                "text_tokenizer": getattr(backbone, "tokenizer_name", None) or args.text_tokenizer,
                "text_tokenizer_resolved": getattr(backbone, "tokenizer_name", None),
                "image_size": args.image_size,
            }
        )

    save_outputs(features, args.output, args.metadata_json)
    return features


def main() -> None:  # pragma: no cover - CLI entry point
    parser = build_argparser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
