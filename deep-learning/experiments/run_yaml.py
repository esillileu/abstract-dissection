"""Run one declarative experiment with the optional MLflow integration."""

from __future__ import annotations

import argparse

from mlprosection_mlflow import run_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an mlprosection YAML experiment.")
    parser.add_argument("config", help="Path to the experiment YAML file")
    parser.add_argument("--atomic-run-id", help="Variant key when the YAML declares variants")
    parser.add_argument("--seed", type=int, help="Master seed from configs/seeds.yaml")
    parser.add_argument("--device", help="Override YAML numerical device, e.g. cuda:0")
    parser.add_argument("--resume", help="Epoch checkpoint directory to resume")
    args = parser.parse_args()
    result = run_yaml(args.config, atomic_run_id=args.atomic_run_id, seed=args.seed, device=args.device, resume=args.resume)
    print(result.metrics)


if __name__ == "__main__":
    main()
