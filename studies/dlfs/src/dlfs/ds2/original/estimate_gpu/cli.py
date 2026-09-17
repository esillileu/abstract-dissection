"""Command line interface for CuPy runtime estimation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .lm import estimate_lstm_rnnlm, estimate_rnnlm
from .schema import BOOK_ROOT, DEFAULT_OUTPUT
from .w2v import estimate_word2vec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate DS2 e02-e04 runtimes through the book's official CuPy path."
    )
    parser.add_argument("--word2vec-updates", type=int, default=100)
    parser.add_argument("--rnnlm-epochs", type=int, default=20)
    parser.add_argument("--lstm-updates", type=int, default=100)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("e02", "e03", "e04"),
        default=("e02", "e03"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.word2vec_updates < 1 or args.rnnlm_epochs < 1 or args.lstm_updates < 1:
        parser.error("benchmark sizes must be positive")

    results = []
    if "e02" in args.experiments:
        results.extend(
            (
                estimate_word2vec(
                    "CBOW",
                    benchmark_updates=args.word2vec_updates,
                ),
                estimate_word2vec(
                    "SkipGram",
                    benchmark_updates=args.word2vec_updates,
                ),
            )
        )
    if "e03" in args.experiments:
        results.append(estimate_rnnlm(benchmark_epochs=args.rnnlm_epochs))
    if "e04" in args.experiments:
        results.append(estimate_lstm_rnnlm(benchmark_updates=args.lstm_updates))
    payload = {
        "method": (
            "Original deep-learning-from-scratch-2 models/trainers using their "
            "official config.GPU=True CuPy path, with common.np bypassed only "
            "because its legacy np.add.at assignment is read-only in CuPy 14. "
            "Native CuPy add.at has the same scatter-add semantics. When e02 is "
            "selected, adapted full-softmax trials are excluded."
        ),
        "book_root": str(BOOK_ROOT),
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for result in results:
        print(
            f"{result.experiment_id}/{result.condition}: "
            f"projected={result.projected_total_time_s:.1f}s",
            flush=True,
        )
    print(f"saved: {args.output}", flush=True)
