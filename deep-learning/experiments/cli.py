"""Unified command-line interface for declared experiment domains."""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml

from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.experiment import load_yaml
from mlprosection_mlflow import run_yaml


EXPERIMENTS_ROOT = Path("experiments")


@dataclass(frozen=True)
class RunPlan:
    domain: str
    experiment_id: str
    path: Path
    atomic_run_id: str
    seed: int | None
    device: str


@dataclass(frozen=True)
class AnalysisPlan:
    domain: str
    experiment_id: str
    module_name: str
    path: Path


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
        parts = key.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"conflicting override path: {key}")
            cursor = child
        cursor[parts[-1]] = yaml.safe_load(raw)
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


def _domain_config_root(domain: str) -> Path:
    root = EXPERIMENTS_ROOT / domain / "config"
    if not root.is_dir():
        raise ValueError(f"unknown experiment domain: {domain}")
    return root


def _seed_values(domain: str, seed_set: str) -> list[int]:
    registry = yaml.safe_load((_domain_config_root(domain) / "seeds.yaml").read_text(encoding="utf-8"))
    try:
        return [int(value) for value in registry["seed_sets"][seed_set]["values"]]
    except KeyError as exc:
        raise ValueError(f"unknown seed set for {domain}: {seed_set}") from exc


def _default_device(domain: str, experiment_id: str) -> str:
    if domain == "deepbase1":
        return "cuda:0" if experiment_id == "e08" else "cpu"
    if domain == "deepbase2":
        return "cuda:0"
    raise ValueError(f"unknown experiment domain: {domain}")


def build_analysis_plans(*, domain: str, experiment_ids: list[str]) -> list[AnalysisPlan]:
    root = EXPERIMENTS_ROOT / domain / "analysis"
    if not root.is_dir():
        raise ValueError(f"analysis directory not found for domain: {domain}")
    selected = {normalize_experiment_id(value) for value in experiment_ids}
    plans = []
    for path in sorted(root.glob("e[0-9][0-9]_*.py")):
        module_name = f"experiments.{domain}.analysis.{path.stem}"
        experiment_id = _analysis_experiment_id(path)
        if experiment_id is None:
            raise ValueError(f"analysis module must declare EXPERIMENT_ID: {module_name}")
        normalized = normalize_experiment_id(experiment_id)
        if selected and normalized not in selected:
            continue
        plans.append(AnalysisPlan(domain=domain, experiment_id=normalized, module_name=module_name, path=path))
    if not plans:
        wanted = ", ".join(sorted(selected))
        raise ValueError(f"no analysis module matched: {wanted}")
    return plans


def _analysis_experiment_id(path: Path) -> str | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "EXPERIMENT_ID" for target in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


def run_analysis_plans(plans: list[AnalysisPlan], *, tracking_uri: str | None, analysis_args: list[str]) -> None:
    for plan in plans:
        command = [sys.executable, "-m", plan.module_name]
        if tracking_uri is not None:
            command.extend(["--tracking-uri", tracking_uri])
        command.extend(analysis_args)
        print(f"{plan.experiment_id} {plan.module_name}")
        subprocess.run(command, check=True)


def build_plans(
    *,
    domain: str,
    experiment_ids: list[str],
    all_experiments: bool,
    seed_set: str,
    seed_indexes: str | None,
    device: str | None,
    overrides: dict[str, object],
) -> list[RunPlan]:
    if bool(experiment_ids) == all_experiments:
        raise ValueError("choose exactly one of --all or --experiment/-e")
    root = _domain_config_root(domain)
    selected = {normalize_experiment_id(value) for value in experiment_ids}
    seeds = _seed_values(domain, seed_set)
    requested_indexes = parse_seed_indexes(seed_indexes, count=len(seeds))
    plans: list[RunPlan] = []
    for path in sorted(root.glob("e[0-9][0-9]_*.yaml")):
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
            available = list(range(count))
            indexes = requested_indexes if requested_indexes is not None else available
            invalid = [index for index in indexes if index not in available]
            if invalid:
                raise ValueError(f"{experiment_id} declares only {count} seed runs; invalid indexes: {invalid}")
            run_seeds = [seeds[index] for index in indexes]
        variants = resolved.get("variants", {})
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"experiment YAML needs variants: {path}")
        for atomic_run_id in variants:
            for seed in run_seeds:
                plans.append(RunPlan(
                    domain=domain,
                    experiment_id=experiment_id,
                    path=path,
                    atomic_run_id=str(atomic_run_id),
                    seed=seed,
                    device=device or str(execution.get("default_device", _default_device(domain, experiment_id))),
                ))
    if not plans:
        wanted = ", ".join(sorted(selected))
        raise ValueError(f"no experiment YAML matched: {wanted}")
    return plans


def _require_mlflow_server(plans: list[RunPlan], overrides: dict[str, object]) -> None:
    uris = set()
    for plan in plans:
        config = load_yaml(plan.path, atomic_run_id=plan.atomic_run_id, overrides=overrides)
        tracking = config.get("tracking", {})
        if isinstance(tracking, dict) and tracking.get("enabled", True):
            uris.add(str(tracking.get("uri", os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))))
    for uri in sorted(uris):
        try:
            with urlopen(f"{uri.rstrip('/')}/health", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"MLflow health check returned HTTP {response.status}: {uri}")
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
        seed = "single" if plan.seed is None else str(plan.seed)
        print(f"{plan.experiment_id} {plan.path.name} {plan.atomic_run_id} seed={seed} device={plan.device}")


def _print_analysis_plans(plans: list[AnalysisPlan]) -> None:
    print(f"{plans[0].domain}: {len(plans)} analysis plans")
    for plan in plans:
        print(f"{plan.experiment_id} {plan.module_name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("deepbase1", "deepbase2"))
    parser.add_argument("command", choices=("plan", "run", "analyze"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true")
    selection.add_argument("-e", "--experiment", action="append", default=[])
    parser.add_argument("--seed-set", default="research_v1")
    parser.add_argument("--seed", help="Seed-set indexes, for example 0-4 or 0,3,7")
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--set", dest="override_values", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--analysis-arg", action="append", default=[], metavar="ARG", help="Argument forwarded to an analysis script; analyze only")
    parser.add_argument("--dry-run", action="store_true", help="Print a run plan without executing it")
    args = parser.parse_args(argv)
    try:
        overrides = parse_overrides(args.override_values)
        all_experiments = args.all or (args.command in {"plan", "analyze"} and not args.experiment)
        if args.command == "run" and not all_experiments and not args.experiment:
            parser.error("run requires --all or --experiment/-e")
        if args.command == "analyze":
            if args.seed or args.device or args.override_values:
                parser.error("analyze does not accept --seed, --device, or --set")
            analysis_plans = build_analysis_plans(
                domain=args.domain,
                experiment_ids=[] if all_experiments else args.experiment,
            )
            _print_analysis_plans(analysis_plans)
            if not args.dry_run:
                run_analysis_plans(analysis_plans, tracking_uri=args.tracking_uri, analysis_args=args.analysis_arg)
            return
        if args.analysis_arg:
            parser.error("--analysis-arg is only valid with analyze")
        plans = build_plans(
            domain=args.domain,
            experiment_ids=args.experiment,
            all_experiments=all_experiments,
            seed_set=args.seed_set,
            seed_indexes=args.seed,
            device=args.device,
            overrides=overrides,
        )
        _print_plans(plans)
        if args.command == "plan" or args.dry_run:
            return
        _require_mlflow_server(plans, overrides)
        _require_devices(plans)
        for plan in plans:
            run_yaml(
                plan.path,
                atomic_run_id=plan.atomic_run_id,
                seed=plan.seed,
                device=plan.device,
                overrides=overrides,
            )
    except ValueError as exc:
        parser.error(str(exc))
