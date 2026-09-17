"""CLI entrypoint for summarizing e05 NVTX operation counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import DEFAULT_INPUT, DEFAULT_MEASUREMENTS, DEFAULT_OUTPUT, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    arguments = parser.parse_args()
    run(arguments.input, arguments.output, arguments.measurements)


if __name__ == "__main__":
    main()
