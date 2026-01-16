"""Convenience wrapper to run :mod:`multimodal_scoring` at 5× only."""

from __future__ import annotations

import argparse
from typing import Sequence

import multimodal_scoring


def build_argparser() -> argparse.ArgumentParser:
    """Create an argument parser that defaults to 5× magnification."""

    parser = multimodal_scoring.build_argparser()
    parser.description = (
        "Evaluate multimodal embeddings while restricting MSCI/CMI-LB computations "
        "to the 5× magnification unless explicitly overridden."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if not args.magnifications:
        args.magnifications = [5]

    return multimodal_scoring.run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
