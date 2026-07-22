"""CLI for declared experiment domains.

The active catalog and its executors live under :mod:`exp`; the historical
``experiments`` package is intentionally not part of this command path.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml

from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.experiment import load_yaml
from mlprosection_mlflow import run_yaml


DOMAIN_ROOT = Path("exp")
DOMAIN_EXECUTOR_MODULES = {"ds1": "exp.ds1.executor", "ds2": "exp.ds2.executor"}


@dataclass(frozen=True)
class RunPlan:
    domain: str
    experiment_id: str
    path: Path
    atomic_run_id: str
    seed: int | None
    device: str


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_overrides(values: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must be KEY=VALUE: {value}")
        key, raw = value.split("=", 1)
        if not key or any(not part for part in key.split(".")):
            raise ValueError(f"override key must use dotted names: {key}")
        cursor = result
        for part in key.split(".")[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"conflicting override path: {key}")
            cursor = child
        cursor[key.split(".")[-1]] = yaml.safe_load(raw)
    return result


def parse_seed_indexes(value: str | None, *, count: int) -> list[int] | None:
    if value is None:
        return None
    indexes: list[int] = []
    for item in value.split(","):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"seed range must be ascending: {item}")
            indexes.extend(range(start, end + 1))
        else:
            indexes.append(int(item))
    selected = list(dict.fromkeys(indexes))
    invalid = [index for index in selected if index < 0 or index >= count]
    if invalid:
        raise ValueError(f"seed indexes out of range 0-{count - 1}: {invalid}")
    return selected


def normalize_experiment_id(value: str) -> str:
    compact = value.lower().removeprefix("e")
    if not compact.isdigit():
        raise ValueError(f"experiment id must be numeric, for example 01: {value}")
    return f"e{int(compact):02d}"


def _config_root(domain: str) -> Path:
    if domain not in DOMAIN_EXECUTOR_MODULES:
        raise ValueError(f"unknown experiment domain: {domain}")
    root = DOMAIN_ROOT / domain / "config"
    if not root.is_dir():
        raise ValueError(f"missing experiment config directory: {root}")
    return root


def _seed_values(domain: str, seed_set: str) -> list[int]:
    registry = yaml.safe_load((_config_root(domain) / "seeds.yaml").read_text(encoding="utf-8"))
    try:
        return [int(value) for value in registry["seed_sets"][seed_set]["values"]]
    except KeyError as exc:
        raise ValueError(f"unknown seed set for {domain}: {seed_set}") from exc


def _default_device(_domain: str, _experiment_id: str) -> str:
    return "cpu"


def build_plans(
    *, domain: str, experiment_ids: list[str], all_experiments: bool,
    seed_set: str, seed_indexes: str | None, device: str | None,
    overrides: dict[str, object],
) -> list[RunPlan]:
    if bool(experiment_ids) == all_experiments:
        raise ValueError("choose exactly one of --all or --experiment/-e")
    selected = {normalize_experiment_id(value) for value in experiment_ids}
    seeds = _seed_values(domain, seed_set)
    requested_indexes = parse_seed_indexes(seed_indexes, count=len(seeds))
    plans: list[RunPlan] = []
    for path in sorted(_config_root(domain).glob("e[0-9][0-9]_*.yaml")):
        experiment_id = path.name[:3]
        if not all_experiments and experiment_id not in selected:
            continue
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError(f"invalid YAML object: {path}")
        resolved = _deep_merge(source, overrides)
        execution = resolved.get("execution", {})
        if not isinstance(execution, dict):
            raise ValueError(f"execution must be a mapping: {path}")
        mode = str(execution.get("mode", "seeded"))
        if mode not in {"seeded", "single"}:
            raise ValueError(f"unsupported execution.mode in {path}: {mode}")
        if mode == "single":
            if requested_indexes is not None:
                raise ValueError(f"{experiment_id} is a single-run experiment and does not accept --seed")
            run_seeds: list[int | None] = [None]
        else:
            policy = resolved.get("policy", {})
            if not isinstance(policy, dict):
                raise ValueError(f"policy must be a mapping: {path}")
            count = int(policy.get("seed_count", len(seeds)))
            indexes = requested_indexes if requested_indexes is not None else list(range(count))
            invalid = [index for index in indexes if index < 0 or index >= count]
            if invalid:
                raise ValueError(f"{experiment_id} declares only {count} seed runs; invalid indexes: {invalid}")
            run_seeds = [seeds[index] for index in indexes]
        variants = resolved.get("variants")
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"experiment YAML needs variants: {path}")
        for atomic_run_id in variants:
            for seed in run_seeds:
                plans.append(RunPlan(domain, experiment_id, path, str(atomic_run_id), seed, device or str(execution.get("default_device", _default_device(domain, experiment_id)))))
    if not plans:
        raise ValueError("no experiment YAML matched")
    return plans


def _require_mlflow_server(plans: list[RunPlan], overrides: dict[str, object]) -> None:
    uris = set()
    for plan in plans:
        config = load_yaml(plan.path, atomic_run_id=plan.atomic_run_id, overrides=overrides)
        tracking = config.get("tracking", {})
        if isinstance(tracking, dict) and tracking.get("enabled", True):
            uris.add(os.getenv("MLFLOW_TRACKING_URI") or str(tracking.get("uri", "http://127.0.0.1:5000")))
    for uri in sorted(uris):
        try:
            with urlopen(f"{uri.rstrip('/')}/health", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"MLflow health check returned HTTP {response.status}")
        except (OSError, URLError) as exc:
            raise RuntimeError(f"MLflow server is unavailable at {uri}. Start it before running plans.") from exc


def _require_devices(plans: list[RunPlan]) -> None:
    for device in sorted({plan.device for plan in plans}):
        try:
            make_backend(BackendConfig(device=device, dtype="float32", seed=0))
        except Exception as exc:
            raise RuntimeError(f"requested device is unavailable: {device}") from exc


def _print_plans(plans: list[RunPlan]) -> None:
    print(f"{plans[0].domain}: {len(plans)} planned runs")
    for plan in plans:
        print(f"{plan.experiment_id} {plan.path.name} {plan.atomic_run_id} seed={'single' if plan.seed is None else plan.seed} device={plan.device}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=tuple(DOMAIN_EXECUTOR_MODULES))
    parser.add_argument("command", choices=("plan", "run"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true")
    selection.add_argument("-e", "--experiment", action="append", default=[])
    parser.add_argument("--seed-set", default="research_v1")
    parser.add_argument("-seed", "--seed", help="Seed-set indexes, for example 0-4 or 0,3,7")
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--set", dest="override_values", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        overrides = parse_overrides(args.override_values)
        all_experiments = args.all or (args.command == "plan" and not args.experiment)
        if args.command == "run" and not all_experiments and not args.experiment:
            parser.error("run requires --all or --experiment/-e")
        plans = build_plans(domain=args.domain, experiment_ids=args.experiment, all_experiments=all_experiments, seed_set=args.seed_set, seed_indexes=args.seed, device=args.device, overrides=overrides)
        _print_plans(plans)
        if args.command == "plan" or args.dry_run:
            return
        _require_mlflow_server(plans, overrides)
        _require_devices(plans)
        for plan in plans:
            run_yaml(plan.path, atomic_run_id=plan.atomic_run_id, seed=plan.seed, device=plan.device, overrides=overrides, executor_module=DOMAIN_EXECUTOR_MODULES[plan.domain])
    except ValueError as exc:
        parser.error(str(exc))
