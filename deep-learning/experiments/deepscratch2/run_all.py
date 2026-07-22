"""Run the declared deepscratch2 sequence-model matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml

from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection_mlflow import run_yaml


CONFIG_ROOT = Path("experiments/deepscratch2/config")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deepscratch2 YAML experiment matrix.")
    parser.add_argument("--experiments", nargs="*", default=None, help="Experiment ids, e.g. e01 e03 e05")
    parser.add_argument("--seed-set", default="research_v1")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plans = list(_plans(args.experiments, _seed_values(args.seed_set), args.device))
    print(f"deepscratch2: {len(plans)} planned runs")
    if not args.dry_run:
        _require_mlflow_server(_tracking_uris(plans))
        _require_device(args.device)
    for path, atomic_run_id, seed, device in plans:
        print(f"{path.name} {atomic_run_id} seed={seed} device={device}")
        if not args.dry_run:
            run_yaml(path, atomic_run_id=atomic_run_id, seed=seed, device=device)


def _seed_values(seed_set: str) -> list[int]:
    registry = yaml.safe_load((CONFIG_ROOT / "seeds.yaml").read_text(encoding="utf-8"))
    return [int(value) for value in registry["seed_sets"][seed_set]["values"]]


def _plans(experiments: list[str] | None, seeds: list[int], device: str):
    selected = set(experiments or ())
    for path in sorted(CONFIG_ROOT.glob("e[0-9][0-9]_*.yaml")):
        experiment_id = path.name[:3]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        run_seeds = seeds[:int(config.get("policy", {}).get("seed_count", len(seeds)))]
        if not run_seeds:
            raise ValueError(f"{path}: policy.seed_count must be at least 1")
        for atomic_run_id, variant in config["variants"].items():
            declared_experiments = set(variant.get("experiment_ids", (experiment_id,)))
            if selected and not selected.intersection(declared_experiments):
                continue
            for seed in run_seeds:
                yield path, atomic_run_id, seed, device


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
        try:
            with urlopen(f"{uri.rstrip('/')}/health", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"MLflow health check returned HTTP {response.status}: {uri}")
        except (OSError, URLError) as exc:
            raise RuntimeError(f"MLflow server is unavailable at {uri}. Start it before running the matrix.") from exc


def _require_device(device: str) -> None:
    try:
        make_backend(BackendConfig(device=device, dtype="float32", seed=0))
    except Exception as exc:
        raise RuntimeError(f"requested device is unavailable: {device}; use --device cpu explicitly") from exc


if __name__ == "__main__":
    main()
