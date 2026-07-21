"""Run the declared deepscratch1 experiment matrix from its YAML catalog."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml

from mlprosection_mlflow import run_yaml


CONFIG_ROOT = Path("experiments/deepscratch1/config")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deepscratch1 YAML experiment matrix.")
    parser.add_argument("--experiments", nargs="*", default=None, help="Experiment ids, e.g. e01 e02 e08")
    parser.add_argument("--seed-set", default="research_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = _seed_values(args.seed_set)
    plans = list(_plans(args.experiments, seeds, args.device))
    print(f"deepscratch1: {len(plans)} planned runs")
    if not args.dry_run:
        _require_mlflow_server(_tracking_uris(plans))
    for path, atomic_run_id, seed, device in plans:
        print(f"{path.name} {atomic_run_id} seed={seed} device={device}")
        if not args.dry_run:
            run_yaml(path, atomic_run_id=atomic_run_id, seed=seed, device=device)


def _seed_values(seed_set: str) -> list[int]:
    registry = yaml.safe_load((CONFIG_ROOT / "seeds.yaml").read_text(encoding="utf-8"))
    values = registry["seed_sets"][seed_set]["values"]
    return [int(value) for value in values]


def _plans(experiments: list[str] | None, seeds: list[int], device_mode: str):
    selected = set(experiments or ())
    for path in sorted(CONFIG_ROOT.glob("e[0-9][0-9]_*.yaml")):
        experiment_id = path.name[:3]
        if selected and experiment_id not in selected:
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        run_seeds = seeds[: int(config.get("policy", {}).get("seed_count", len(seeds)))]
        if not run_seeds:
            raise ValueError(f"{path}: policy.seed_count must be at least 1")
        for atomic_run_id in config["variants"]:
            device = _device_for(experiment_id, device_mode)
            for seed in run_seeds:
                yield path, atomic_run_id, seed, device


def _device_for(experiment_id: str, mode: str) -> str:
    if mode != "auto":
        return mode
    return "cuda:0" if experiment_id == "e08" else "cpu"


def _tracking_uris(plans) -> set[str]:
    uris = set()
    for path, _, _, _ in plans:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        tracking = config.get("tracking", {})
        if tracking.get("enabled", True):
            uris.add(str(tracking.get("uri", os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))))
    return uris


def _require_mlflow_server(uris: set[str]) -> None:
    for uri in sorted(uris):
        health_url = f"{uri.rstrip('/')}/health"
        try:
            with urlopen(health_url, timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"MLflow health check returned HTTP {response.status}: {health_url}")
        except (OSError, URLError) as exc:
            raise RuntimeError(
                f"MLflow server is unavailable at {uri}. Start it before running the matrix."
            ) from exc


if __name__ == "__main__":
    main()
