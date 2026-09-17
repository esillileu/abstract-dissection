from __future__ import annotations

import argparse

from dlfs.ds1.implemented.final_gap import TARGET_RUNS

from .backfill import TARGET_NAMES, backfill_runs, latest_target_runs
from .evaluation import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill DS1 full-train accuracy and train/test gap metrics in MLflow."
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=sorted(TARGET_NAMES),
        help="Repeat to select e06, e07, or e12; defaults to all.",
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from mlflow.tracking import MlflowClient

    from dlfs.tracking import resolve_tracking_uri

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    targets = (
        {TARGET_NAMES[name] for name in args.target}
        if args.target
        else set(TARGET_RUNS)
    )
    runs = latest_target_runs(
        client,
        targets=targets,
        seeds=set(args.seed or ()),
    )
    counts = backfill_runs(
        client,
        runs,
        apply=args.apply,
        force=args.force,
        evaluator=lambda run: evaluate_run(client, run, device=args.device),
    )
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
