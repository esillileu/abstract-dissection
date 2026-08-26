"""Temporary CLI for evaluating the book's pretrained BetterRnnlm weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlfs.ds2.original.run.e05 import evaluate_pretrained

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "BetterRnnlm (1).pkl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained BetterRnnlm weights on the full PTB test split."
    )
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--device", default="cuda:0", help="cpu or a CUDA device such as cuda:0"
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--time-size", type=int, default=35)
    args = parser.parse_args()

    perplexity = evaluate_pretrained(
        SOURCE_ROOT,
        args.checkpoint,
        selected_device=args.device,
        batch_size=args.batch_size,
        time_size=args.time_size,
    )
    print(f"full test perplexity: {perplexity:.6f}")


if __name__ == "__main__":
    main()
