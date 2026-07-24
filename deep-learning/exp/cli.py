"""CLI for declared experiment domains.

The active catalog and its executors live under :mod:`exp`; the historical
``experiments`` package is intentionally not part of this command path.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml

from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.experiment.progress import ProgressManager, RunProgressContext
from mlprosection_mlflow import run_yaml


DOMAIN_ROOT = Path("exp")
DOMAIN_EXECUTOR_MODULES = {"ds1": "exp.ds1.executor", "ds2": "exp.ds2.executor"}
DOMAIN_SPEC_MODULES = {"ds1": "exp.ds1.spec", "ds2": "exp.ds2.spec"}
DOMAIN_ANALYSIS_MODULES = {"ds1": "exp.ds1.analyze.render", "ds2": "exp.ds2.analyze.render"}


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


def parse_experiment_ids(values: list[str]) -> list[str]:
    """Expand 01, e01, 01-08, and comma-separated execution selections."""
    selected: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip().lower()
            match = re.fullmatch(r"e?(\d+)(?:-e?(\d+))?", item)
            if match is None:
                raise ValueError(f"invalid experiment selection: {item}")
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
            if start > end:
                raise ValueError(f"experiment range must be ascending: {item}")
            selected.extend(f"e{number:02d}" for number in range(start, end + 1))
    return list(dict.fromkeys(selected))


def parse_atomic_run_ids(values: list[str]) -> list[str]:
    """Parse repeatable, comma-separated atomic run ID selections."""
    selected: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                raise ValueError("atomic run ID must not be empty")
            selected.append(item)
    return list(dict.fromkeys(selected))


def _config_root(domain: str) -> Path:
    if domain not in DOMAIN_EXECUTOR_MODULES:
        raise ValueError(f"unknown experiment domain: {domain}")
    root = DOMAIN_ROOT / domain / "config"
    if not root.is_dir():
        raise ValueError(f"missing experiment config directory: {root}")
    return root


def parse_domain_run_spec(domain: str, path: Path, *, atomic_run_id: str | None = None, overrides: dict[str, object] | None = None):
    if domain not in DOMAIN_SPEC_MODULES:
        raise ValueError(f"unknown experiment domain: {domain}")
    module = importlib.import_module(DOMAIN_SPEC_MODULES[domain])
    return module.parse_run_spec(path, atomic_run_id=atomic_run_id, overrides=overrides)


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
    atomic_run_ids: list[str] | None = None,
    excluded_atomic_run_ids: list[str] | None = None,
    seed_first: bool = False,
) -> list[RunPlan]:
    if bool(experiment_ids) == all_experiments:
        raise ValueError("choose exactly one of --all or --experiment/-e")
    if atomic_run_ids and excluded_atomic_run_ids:
        raise ValueError("choose at most one of --atomic-run/-a or --exclude-atomic-run/-x")
    selected = set(parse_experiment_ids(experiment_ids))
    included_atomic_runs = set(parse_atomic_run_ids(atomic_run_ids or []))
    excluded_atomic_runs = set(parse_atomic_run_ids(excluded_atomic_run_ids or []))
    requested_atomic_runs = included_atomic_runs or excluded_atomic_runs
    matched_atomic_runs: set[str] = set()
    seeds = _seed_values(domain, seed_set)
    requested_indexes = parse_seed_indexes(seed_indexes, count=len(seeds))
    ordered_seed_indexes = (
        requested_indexes
        if requested_indexes is not None
        else list(range(len(seeds)))
    )
    seed_order = {
        seeds[index]: position
        for position, index in enumerate(ordered_seed_indexes)
    }
    plans: list[RunPlan] = []
    for path in sorted(_config_root(domain).glob("e[0-9][0-9]_*.yaml")):
        experiment_id = path.name[:3]
        if not all_experiments and experiment_id not in selected:
            continue
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError(f"invalid YAML object: {path}")
        resolved = _deep_merge(source, overrides)
        variants = resolved.get("variants")
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"experiment YAML needs variants: {path}")
        available_atomic_runs = {str(atomic_run_id) for atomic_run_id in variants}
        matched_atomic_runs.update(requested_atomic_runs & available_atomic_runs)
        selected_variants = [
            str(atomic_run_id)
            for atomic_run_id in variants
            if (
                not included_atomic_runs
                or str(atomic_run_id) in included_atomic_runs
            )
            and str(atomic_run_id) not in excluded_atomic_runs
        ]
        if not selected_variants:
            continue
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
            policy = resolved.get("seed_policy", {})
            if not isinstance(policy, dict):
                raise ValueError(f"seed_policy must be a mapping: {path}")
            count = int(policy.get("seed_count", len(seeds)))
            indexes = requested_indexes if requested_indexes is not None else list(range(count))
            invalid = [index for index in indexes if index < 0 or index >= count]
            if invalid:
                raise ValueError(f"{experiment_id} declares only {count} seed runs; invalid indexes: {invalid}")
            run_seeds = [seeds[index] for index in indexes]
        for atomic_run_id in selected_variants:
            for seed in run_seeds:
                plans.append(RunPlan(domain, experiment_id, path, str(atomic_run_id), seed, device or str(execution.get("default_device", _default_device(domain, experiment_id)))))
    unknown_atomic_runs = requested_atomic_runs - matched_atomic_runs
    if unknown_atomic_runs:
        unknown = ", ".join(sorted(unknown_atomic_runs))
        raise ValueError(f"unknown atomic run ID in selected experiments: {unknown}")
    if not plans:
        if requested_atomic_runs:
            raise ValueError("atomic run selection matched no plans")
        raise ValueError("no experiment YAML matched")
    if seed_first:
        plans.sort(
            key=lambda plan: (
                seed_order.get(plan.seed, len(seed_order))
                if plan.seed is not None
                else len(seed_order)
            )
        )
    return plans


def _require_mlflow_server(plans: list[RunPlan], overrides: dict[str, object]) -> None:
    uris = set()
    for plan in plans:
        config = parse_domain_run_spec(plan.domain, plan.path, atomic_run_id=plan.atomic_run_id, overrides=overrides).to_executor_config()
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


def _progress_context(plan: RunPlan, *, index: int, count: int, overrides: dict[str, object]) -> RunProgressContext:
    config = parse_domain_run_spec(
        plan.domain,
        plan.path,
        atomic_run_id=plan.atomic_run_id,
        overrides=overrides,
    ).to_executor_config()
    training = config.get("training", {})
    total_updates = None
    if isinstance(training, dict) and training.get("max_updates") is not None:
        total_updates = int(training["max_updates"])
    return RunProgressContext(
        label=f"{plan.experiment_id}/{plan.atomic_run_id}/s{'single' if plan.seed is None else plan.seed}",
        index=index,
        count=count,
        total_updates=total_updates,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=tuple(DOMAIN_EXECUTOR_MODULES))
    parser.add_argument("command", choices=("plan", "run", "analyze"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true")
    selection.add_argument("-e", "--experiment", action="append", default=[])
    atomic_selection = parser.add_mutually_exclusive_group()
    atomic_selection.add_argument(
        "-a",
        "--atomic-run",
        action="append",
        default=[],
        help="Run only these atomic IDs; repeat or separate with commas",
    )
    atomic_selection.add_argument(
        "-x",
        "--exclude-atomic-run",
        action="append",
        default=[],
        help="Exclude these atomic IDs; repeat or separate with commas",
    )
    parser.add_argument("--seed-set", default="research_v1")
    parser.add_argument("-seed", "--seed", help="Seed-set indexes, for example 0-4 or 0,3,7")
    parser.add_argument(
        "--seed-first",
        action="store_true",
        help="Run all selected atomic runs for each seed before moving to the next seed",
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--set", dest="override_values", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress", choices=("auto", "none", "line", "tqdm"), default="auto")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--error-style", choices=("band", "errorbar"), default="band")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Print final metric and training-time summaries for analyze",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            if args.atomic_run or args.exclude_atomic_run or args.seed_first:
                parser.error(
                    "atomic run selection and --seed-first are supported only for plan and run"
                )
            analysis_argv: list[str] = []
            if args.all:
                analysis_argv.extend(("-e", "all"))
            for experiment in args.experiment:
                analysis_argv.extend(("-e", experiment))
            if args.tracking_uri:
                analysis_argv.extend(("--tracking-uri", args.tracking_uri))
            analysis_argv.extend(("--error-style", args.error_style))
            if args.output_dir is not None:
                analysis_argv.extend(("--output-dir", str(args.output_dir)))
            if args.seed is not None:
                analysis_argv.extend(("--seed", args.seed))
            if args.summary:
                analysis_argv.append("--summary")
            importlib.import_module(DOMAIN_ANALYSIS_MODULES[args.domain]).main(analysis_argv)
            return
        overrides = parse_overrides(args.override_values)
        all_experiments = args.all or (args.command == "plan" and not args.experiment)
        if args.command == "run" and not all_experiments and not args.experiment:
            parser.error("run requires --all or --experiment/-e")
        plans = build_plans(
            domain=args.domain,
            experiment_ids=args.experiment,
            all_experiments=all_experiments,
            seed_set=args.seed_set,
            seed_indexes=args.seed,
            device=args.device,
            overrides=overrides,
            atomic_run_ids=args.atomic_run,
            excluded_atomic_run_ids=args.exclude_atomic_run,
            seed_first=args.seed_first,
        )
        _print_plans(plans)
        if args.command == "plan" or args.dry_run:
            return
        _require_mlflow_server(plans, overrides)
        _require_devices(plans)
        progress = ProgressManager(
            mode=args.progress,
            every=args.progress_every,
            total_runs=len(plans),
        )
        try:
            for index, plan in enumerate(plans, start=1):
                progress_context = _progress_context(plan, index=index, count=len(plans), overrides=overrides)
                reporter = progress.reporter(progress_context)
                progress.on_run_start(progress_context)
                try:
                    run_yaml(
                        plan.path,
                        atomic_run_id=plan.atomic_run_id,
                        seed=plan.seed,
                        device=plan.device,
                        overrides=overrides,
                        executor_module=DOMAIN_EXECUTOR_MODULES[plan.domain],
                        spec_module=DOMAIN_SPEC_MODULES[plan.domain],
                        progress_reporter=reporter,
                    )
                finally:
                    reporter.close()
                    progress.on_run_end()
        finally:
            progress.close()
    except ValueError as exc:
        parser.error(str(exc))
