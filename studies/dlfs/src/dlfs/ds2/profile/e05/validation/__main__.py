"""CLI entrypoint for TimeLSTM correctness validation."""

import argparse
from pathlib import Path

from ..benchmark import DEFAULT_RESULTS
from .runner import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--stage", choices=("phase1", "phase2", "phase3"), default="phase1"
    )
    arguments = parser.parse_args()
    run(arguments.output_dir, stage=arguments.stage)
