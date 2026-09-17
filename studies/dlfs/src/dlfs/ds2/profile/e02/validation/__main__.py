"""CLI entrypoint for e02 fused Word2Vec validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import DEFAULT_OUTPUT, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--device", action="append", default=None)
    args = parser.parse_args()
    devices = tuple(args.device or ("cpu",))
    result = run(args.output, devices=devices, report=args.report)
    print(json.dumps(result, indent=2))
    print(f"JSON report: {args.output}")
    print(f"Markdown report: {args.report or args.output.with_suffix('.md')}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
