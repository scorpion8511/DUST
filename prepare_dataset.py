"""Utilities to prepare multi-magnification features from BreakHis-style datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

# Magnifications represented in the dataset.  Corresponding directories may be
# named e.g. ``40X`` or ``40x``; the code matches case-insensitively.
MAGNIFICATIONS = ["40", "100", "200", "400"]


def _image_histogram(path: Path) -> np.ndarray:
    """Return a normalised RGB histogram for an image file."""
    with Image.open(path) as img:
        hist = np.array(img.convert("RGB").histogram(), dtype=np.float32)
    # normalise histogram to sum to one
    return hist / hist.sum() if hist.sum() else hist


def prepare_embeddings(data_root: Path, output_root: Path, dataset: str = "SOB", feature_name: str = "histogram") -> None:
    """Prepare per-magnification embedding files from a nested image dataset.

    The expected directory structure is::

        data_root/
            cancer_type/
                region_id/
                    40X/ 100X/ 200X/ 400X/   # magnification folders
                        *.png                 # images at this magnification

    Features are computed as average colour histograms for all images in a
    region/magnification.  The resulting arrays are saved to
    ``output_root/dataset/feature_name/mag{mag}.npy`` with shape
    ``(n_regions, dim)``.
    """
    features: Dict[str, List[np.ndarray]] = {mag: [] for mag in MAGNIFICATIONS}

    region_dirs: List[Path] = []
    for cancer_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for region_dir in sorted(p for p in cancer_dir.iterdir() if p.is_dir()):
            region_dirs.append(region_dir)
            for mag in MAGNIFICATIONS:
                # locate the magnification directory, allowing optional and
                # case-insensitive ``x`` suffix (e.g. ``40X`` or ``40x``)
                candidates = [
                    region_dir / mag,
                    region_dir / f"{mag}x",
                    region_dir / f"{mag}X",
                ]
                img_dir = next((p for p in candidates if p.is_dir()), None)
                if img_dir is None:
                    raise FileNotFoundError(
                        f"no images for {region_dir} at {mag}X"
                    )

                imgs = sorted(img_dir.glob("*.png"))
                if not imgs:
                    raise FileNotFoundError(
                        f"no images for {region_dir} at {mag}X"
                    )
                feats = [_image_histogram(img) for img in imgs]
                features[mag].append(np.mean(feats, axis=0))

    out_dir = output_root / dataset / feature_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for mag, feats in features.items():
        arr = np.stack(feats, axis=0)
        np.save(out_dir / f"mag{mag}.npy", arr)

    # optionally record region order for reference
    (out_dir / "regions.txt").write_text("\n".join(r.name for r in region_dirs))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare multi-magnification features")
    parser.add_argument("data_root", type=Path, help="Root of BreakHis-style dataset")
    parser.add_argument("output_root", type=Path, help="Directory to store features")
    parser.add_argument("--dataset", default="SOB", help="Dataset name for output path")
    parser.add_argument(
        "--feature-name",
        default="histogram",
        help="Name of feature directory under dataset",
    )
    args = parser.parse_args()

    prepare_embeddings(args.data_root, args.output_root, args.dataset, args.feature_name)
