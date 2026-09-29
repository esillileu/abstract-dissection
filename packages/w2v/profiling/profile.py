#!/usr/bin/env python3
"""Reproducible perf harness for the three w2v implementations."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
BUILD = HERE / "build"
DATA = HERE / "data"
RESULTS = HERE / "results"
EVENTS = (
    "duration_time,task-clock,cycles,instructions,branches,branch-misses,"
    "cache-references,cache-misses,context-switches,cpu-migrations,page-faults"
)
MODELS = ("cbow", "skipgram")
OBJECTIVES = ("hs", "negative")


def run(
    command: list[str], *, capture: bool = True, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PACKAGE,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_config(mode: str, scale: float) -> dict[str, object]:
    if mode == "smoke":
        return {
            "seed": 1,
            "lexical_vocabulary": 1000,
            "tokens": 20_000,
            "sentence_tokens": 50,
            "distribution": "zipf-1",
        }
    return {
        "seed": 1,
        "lexical_vocabulary": 50_000,
        "tokens": round(2_000_000 * scale),
        "sentence_tokens": 50,
        "distribution": "zipf-1",
    }


def generate_corpus(config: dict[str, object]) -> tuple[Path, str, str]:
    DATA.mkdir(parents=True, exist_ok=True)
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    path = DATA / f"corpus-{config_hash[:16]}.txt"
    metadata = path.with_suffix(".json")
    if path.exists() and metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved.get("config_sha256") == config_hash and saved.get(
            "corpus_sha256"
        ) == sha256(path):
            return path, saved["corpus_sha256"], config_hash

    vocabulary = int(config["lexical_vocabulary"])
    token_count = int(config["tokens"])
    sentence_tokens = int(config["sentence_tokens"])
    minimum = 5 * vocabulary
    if token_count < minimum:
        raise SystemExit(
            f"token count {token_count} cannot give {vocabulary} words min-count 5"
        )
    harmonic = 0.0
    cumulative: list[float] = []
    for rank in range(1, vocabulary + 1):
        harmonic += 1.0 / rank
        cumulative.append(harmonic)
    state = int(config["seed"])
    emitted = 0
    with path.open("w", encoding="ascii", buffering=1024 * 1024) as stream:

        def emit(index: int) -> None:
            nonlocal emitted
            stream.write(f"w{index:05d}")
            emitted += 1
            stream.write("\n" if emitted % sentence_tokens == 0 else " ")

        # Establish exactly the requested retained vocabulary before Zipf draws.
        for _ in range(5):
            for index in range(vocabulary):
                emit(index)
        for _ in range(token_count - minimum):
            state = (state * 25214903917 + 11) & ((1 << 64) - 1)
            point = ((state >> 11) / float(1 << 53)) * harmonic
            emit(bisect.bisect_left(cumulative, point))
        if emitted % sentence_tokens:
            stream.write("\n")
    corpus_hash = sha256(path)
    metadata.write_text(
        json.dumps(
            {
                "config": config,
                "config_sha256": config_hash,
                "corpus_sha256": corpus_hash,
            },
            indent=2,
        )
        + "\n"
    )
    return path, corpus_hash, config_hash


def cpu_order() -> tuple[list[int], list[dict[str, int]]]:
    allowed = sorted(os.sched_getaffinity(0))
    groups: dict[tuple[int, int], list[int]] = {}
    topology: list[dict[str, int]] = []
    for cpu in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((base / "physical_package_id").read_text())
            core = int((base / "core_id").read_text())
        except (OSError, ValueError):
            package, core = 0, cpu
        groups.setdefault((package, core), []).append(cpu)
        topology.append({"cpu": cpu, "package": package, "core": core})
    ordered = [siblings[0] for siblings in groups.values()]
    ordered.extend(cpu for siblings in groups.values() for cpu in siblings[1:])
    return ordered, topology


def command_for(
    implementation: str,
    corpus: Path,
    model: str,
    objective: str,
    threads: int,
    epochs: int,
    update_strategy: str,
) -> list[str]:
    if implementation == "original":
        # The immutable upstream CLI stores -train in a 100-byte array.
        # Keep the harness path short without changing that snapshot.
        corpus_argument = str(corpus.relative_to(PACKAGE))
        return [
            str(BUILD / "original-w2v"),
            "-train",
            corpus_argument,
            "-output",
            "/dev/null",
            "-size",
            "100",
            "-window",
            "5",
            "-sample",
            "0",
            "-negative",
            "5" if objective == "negative" else "0",
            "-hs",
            "1" if objective == "hs" else "0",
            "-threads",
            str(threads),
            "-iter",
            str(epochs),
            "-min-count",
            "5",
            "-cbow",
            "1" if model == "cbow" else "0",
            "-debug",
            "0",
            "-binary",
            "1",
        ]
    executable = BUILD / (
        "reference-w2v" if implementation == "reference" else "rust-w2v"
    )
    command = [
        str(executable),
        str(corpus),
        model,
        objective,
        str(threads),
        str(epochs),
    ]
    if implementation == "rust":
        command.extend(("100000000", update_strategy))
    return command


def parse_perf(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as stream:
        for fields in csv.reader(stream, delimiter=";"):
            if len(fields) < 3 or not fields[2].strip():
                continue
            value = fields[0].strip().replace(" ", "")
            if value.startswith("<"):
                raise RuntimeError(f"invalid perf counter {fields[2].strip()}: {value}")
            try:
                numeric = float(value)
            except ValueError:
                continue
            event = fields[2].strip().split(":", 1)[0]
            rows.append({"event": event, "value": numeric, "unit": fields[1].strip()})
    expected = set(EVENTS.split(","))
    present = {str(row["event"]) for row in rows}
    missing = expected - present
    if missing:
        raise RuntimeError(f"missing perf counters in {path}: {sorted(missing)}")
    return rows


_RUNNER_METRIC_PATTERN = re.compile(
    r"^(training_seconds|processed_tokens|processed_tokens_per_training_second)=(.+)$"
)


def runner_metrics(output: str, implementation: str) -> dict[str, float | int] | None:
    """Parse and validate the training-only metrics emitted by modular runners."""
    if implementation == "original":
        return None
    values: dict[str, str] = {}
    for part in output.split():
        match = _RUNNER_METRIC_PATTERN.match(part)
        if match:
            values[match.group(1)] = match.group(2)
    required = (
        "training_seconds",
        "processed_tokens",
        "processed_tokens_per_training_second",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(
            f"{implementation} runner missing training metrics: {missing}"
        )
    try:
        training_seconds = float(values["training_seconds"])
        processed = int(values["processed_tokens"])
        tokens_per_second = float(values["processed_tokens_per_training_second"])
    except ValueError as error:
        raise RuntimeError(f"invalid {implementation} training metric") from error
    if (
        not all(
            math.isfinite(value) and value > 0
            for value in (training_seconds, tokens_per_second)
        )
        or processed <= 0
    ):
        raise RuntimeError(f"invalid {implementation} training metric values")
    return {
        "training_seconds": training_seconds,
        "processed_tokens": processed,
        "training_tokens_per_second": tokens_per_second,
    }


def stat_sample(
    output_dir: Path,
    implementation: str,
    corpus: Path,
    model: str,
    objective: str,
    threads: int,
    repetition: int,
    cpus: list[int],
    update_strategy: str,
) -> tuple[list[dict[str, object]], dict[str, float | int] | None]:
    stem = f"{implementation}-{model}-{objective}-t{threads}-r{repetition}"
    raw_path = output_dir / "stat" / f"{stem}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "perf",
        "stat",
        "--no-big-num",
        "-x;",
        "-e",
        EVENTS,
        "-o",
        str(raw_path),
        "--",
        "taskset",
        "-c",
        ",".join(map(str, cpus[:threads])),
        *command_for(
            implementation, corpus, model, objective, threads, 1, update_strategy
        ),
    ]
    completed = run(command, check=False)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "logs" / f"{stem}.stdout").write_text(completed.stdout)
    (output_dir / "logs" / f"{stem}.stderr").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{stem} exited {completed.returncode}; see logs")
    metrics = runner_metrics(completed.stdout, implementation)
    actual = None if metrics is None else int(metrics["processed_tokens"])
    rows = parse_perf(raw_path)
    duration = next(
        float(row["value"]) / 1e9 for row in rows if row["event"] == "duration_time"
    )
    if metrics is not None and float(metrics["training_seconds"]) > duration:
        raise RuntimeError(f"{stem} training time exceeds whole-process time")
    for row in rows:
        row.update(
            implementation=implementation,
            model=model,
            objective=objective,
            threads=threads,
            repetition=repetition,
            processed_tokens=actual,
        )
    if metrics is not None:
        metrics.update(
            implementation=implementation,
            model=model,
            objective=objective,
            threads=threads,
            repetition=repetition,
        )
    return rows, metrics


def write_summaries(
    output_dir: Path,
    rows: list[dict[str, object]],
    metrics: list[dict[str, float | int]],
    nominal_tokens: int,
) -> None:
    fields = [
        "implementation",
        "model",
        "objective",
        "threads",
        "repetition",
        "event",
        "value",
        "unit",
        "processed_tokens",
    ]
    with (output_dir / "raw_stats.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[object, ...], list[float]] = {}
    units: dict[tuple[object, ...], object] = {}
    for row in rows:
        key = tuple(
            row[name]
            for name in ("implementation", "model", "objective", "threads", "event")
        )
        groups.setdefault(key, []).append(float(row["value"]))
        units[key] = row["unit"]
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "implementation",
                "model",
                "objective",
                "threads",
                "event",
                "unit",
                "median",
                "mad",
            ]
        )
        for key, values in sorted(groups.items()):
            median = statistics.median(values)
            writer.writerow(
                [
                    *key,
                    units[key],
                    median,
                    statistics.median(abs(value - median) for value in values),
                ]
            )
    durations = {
        key[:4]: statistics.median(values)
        for key, values in groups.items()
        if key[4] == "duration_time"
    }
    with (output_dir / "speedup.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "implementation",
                "comparison_class",
                "model",
                "objective",
                "threads",
                "wall_seconds",
                "nominal_tokens_per_second",
                "speedup_vs_1_thread",
            ]
        )
        for key, nanoseconds in sorted(durations.items()):
            implementation, model, objective, threads = key
            seconds = nanoseconds / 1e9
            writer.writerow(
                [
                    implementation,
                    "upstream-baseline"
                    if implementation == "original"
                    else "atomic-contract",
                    model,
                    objective,
                    threads,
                    seconds,
                    nominal_tokens / seconds,
                    durations[(implementation, model, objective, 1)] / nanoseconds,
                ]
            )
    metric_fields = [
        "implementation",
        "model",
        "objective",
        "threads",
        "repetition",
        "training_seconds",
        "processed_tokens",
        "training_tokens_per_second",
    ]
    with (output_dir / "runner_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics)
    grouped: dict[tuple[object, ...], list[float]] = {}
    for metric in metrics:
        key_prefix = tuple(
            metric[name] for name in ("implementation", "model", "objective", "threads")
        )
        for name in (
            "training_seconds",
            "processed_tokens",
            "training_tokens_per_second",
        ):
            grouped.setdefault((*key_prefix, name), []).append(float(metric[name]))
    with (output_dir / "training_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "implementation",
                "model",
                "objective",
                "threads",
                "metric",
                "unit",
                "median",
                "mad",
            ]
        )
        units = {
            "training_seconds": "seconds",
            "processed_tokens": "tokens",
            "training_tokens_per_second": "tokens_per_second",
        }
        for key, values in sorted(grouped.items()):
            median = statistics.median(values)
            writer.writerow(
                [
                    *key[:-1],
                    key[-1],
                    units[key[-1]],
                    median,
                    statistics.median(abs(value - median) for value in values),
                ]
            )


def profile_stacks(
    output_dir: Path,
    corpus: Path,
    implementations: tuple[str, ...],
    models: tuple[str, ...],
    objectives: tuple[str, ...],
    cpus: list[int],
    update_strategy: str,
) -> int:
    count = 0
    stack_dir = output_dir / "stacks"
    stack_dir.mkdir(exist_ok=True)
    for implementation in implementations:
        for model in models:
            for objective in objectives:
                for threads in (1, 12):
                    command = [
                        "taskset",
                        "-c",
                        ",".join(map(str, cpus[:threads])),
                        *command_for(
                            implementation,
                            corpus,
                            model,
                            objective,
                            threads,
                            1,
                            update_strategy,
                        ),
                    ]
                    started = time.monotonic()
                    pilot = run(command, check=False)
                    elapsed = time.monotonic() - started
                    if pilot.returncode != 0:
                        raise RuntimeError(
                            f"stack pilot failed: {implementation}/{model}/{objective}/t{threads}: {pilot.stderr}"
                        )
                    epochs = max(1, math.ceil(3.0 / max(elapsed, 0.001)))
                    stem = f"{implementation}-{model}-{objective}-t{threads}"
                    data_path = stack_dir / f"{stem}.perf.data"
                    record = [
                        "perf",
                        "record",
                        "-g",
                        "--call-graph",
                        "fp",
                        "-o",
                        str(data_path),
                        "--",
                        "taskset",
                        "-c",
                        ",".join(map(str, cpus[:threads])),
                        *command_for(
                            implementation,
                            corpus,
                            model,
                            objective,
                            threads,
                            epochs,
                            update_strategy,
                        ),
                    ]
                    completed = run(record, check=False)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"stack recording failed for {stem}: {completed.stderr}"
                        )
                    report = run(
                        [
                            "perf",
                            "report",
                            "--stdio",
                            "--no-children",
                            "-i",
                            str(data_path),
                        ],
                        check=False,
                    )
                    if report.returncode != 0 or not report.stdout.strip():
                        raise RuntimeError(
                            f"perf report failed for {stem}: {report.stderr}"
                        )
                    (stack_dir / f"{stem}.report.txt").write_text(report.stdout)
                    count += 1
    return count


def version(command: list[str]) -> str:
    completed = run(command, check=False)
    return (
        (completed.stdout or completed.stderr).splitlines()[0]
        if (completed.stdout or completed.stderr)
        else "unavailable"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "run"))
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="explicit multiplier for the 2M-token full corpus",
    )
    parser.add_argument(
        "--threads",
        default=None,
        help="comma-separated thread subset for stat samples; default keeps the existing matrix",
    )
    parser.add_argument(
        "--models", default=",".join(MODELS), help="comma-separated model subset"
    )
    parser.add_argument(
        "--objectives",
        default=",".join(OBJECTIVES),
        help="comma-separated objective subset",
    )
    parser.add_argument(
        "--update-strategy", choices=("cas", "hogwild"), default="hogwild"
    )
    parser.add_argument(
        "--build-id",
        default="rust-release-thin-lto",
        help="build identity recorded in the manifest",
    )
    args = parser.parse_args()
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive")
    for executable in ("make", "cargo", "perf", "taskset", "git", "cc"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable not found: {executable}")
    probe = run(["perf", "stat", "-e", EVENTS, "--", "true"], check=False)
    if (
        probe.returncode != 0
        or "not supported" in probe.stderr
        or "not counted" in probe.stderr
    ):
        raise SystemExit(f"required perf counters are unavailable:\n{probe.stderr}")
    run(["make", "-C", str(HERE), "all"], capture=False)
    config = corpus_config(args.mode, args.scale)
    corpus, corpus_hash, config_hash = generate_corpus(config)
    cpus, topology = cpu_order()
    default_thread_counts = (1, 2) if args.mode == "smoke" else (1, 2, 4, 6, 12)
    if args.threads is None:
        thread_counts = default_thread_counts
    else:
        try:
            thread_counts = tuple(
                dict.fromkeys(int(value) for value in args.threads.split(","))
            )
        except ValueError:
            parser.error(
                "--threads must be a comma-separated list of positive integers"
            )
        if not thread_counts or any(value <= 0 for value in thread_counts):
            parser.error(
                "--threads must be a comma-separated list of positive integers"
            )
    if len(cpus) < max(thread_counts):
        raise SystemExit(
            f"need {max(thread_counts)} available logical CPUs, found {len(cpus)}"
        )
    try:
        models = tuple(dict.fromkeys(args.models.split(",")))
        objectives = tuple(dict.fromkeys(args.objectives.split(",")))
    except AttributeError:
        parser.error("--models and --objectives must be comma-separated lists")
    if not models or any(model not in MODELS for model in models):
        parser.error(f"--models must contain only: {','.join(MODELS)}")
    if not objectives or any(objective not in OBJECTIVES for objective in objectives):
        parser.error(f"--objectives must contain only: {','.join(OBJECTIVES)}")
    repetitions = 1 if args.mode == "smoke" else 3
    implementations = ("original", "reference", "rust")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = RESULTS / f"{args.mode}-{stamp}"
    output_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    metrics: list[dict[str, float | int]] = []
    for implementation in implementations:
        for model in models:
            for objective in objectives:
                for threads in thread_counts:
                    for repetition in range(1, repetitions + 1):
                        print(
                            f"stat {implementation} {model}/{objective} t={threads} r={repetition}",
                            flush=True,
                        )
                        sample_rows, sample_metrics = stat_sample(
                            output_dir,
                            implementation,
                            corpus,
                            model,
                            objective,
                            threads,
                            repetition,
                            cpus,
                            args.update_strategy,
                        )
                        rows.extend(sample_rows)
                        if sample_metrics is not None:
                            metrics.append(sample_metrics)
    expected_samples = (
        len(implementations)
        * len(models)
        * len(objectives)
        * len(thread_counts)
        * repetitions
    )
    observed_samples = len(
        {
            (
                row["implementation"],
                row["model"],
                row["objective"],
                row["threads"],
                row["repetition"],
            )
            for row in rows
        }
    )
    if observed_samples != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} stat samples, found {observed_samples}"
        )
    write_summaries(output_dir, rows, metrics, int(config["tokens"]))
    profile_count = (
        0
        if args.mode == "smoke"
        else profile_stacks(
            output_dir,
            corpus,
            implementations,
            models,
            objectives,
            cpus,
            args.update_strategy,
        )
    )
    expected_profiles = len(implementations) * len(models) * len(objectives) * 2
    if args.mode == "run" and profile_count != expected_profiles:
        raise RuntimeError(
            f"expected {expected_profiles} stack profiles, found {profile_count}"
        )
    duration_values = [
        float(row["value"]) / 1e9 for row in rows if row["event"] == "duration_time"
    ]
    manifest = {
        "schema": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "interpretation": {
            "timing_scope": "whole-process wall time in perf; modular runner training_seconds covers trainer.train() only",
            "throughput": "nominal input corpus tokens per wall second",
            "original": "upstream baseline with different Skip-gram, RNG/learning-rate, and Hogwild semantics",
            "reference_and_rust": "direct comparison under the shared algorithm and atomic-update contract",
            "original_training_only": "unavailable; immutable upstream runner output is unchanged",
        },
        "workload": {
            **config,
            "dimension": 100,
            "window": 5,
            "min_count": 5,
            "dynamic_window": True,
            "rng": "lcg",
            "negative_samples": 5,
            "negative_table_size": 100_000_000,
            "subsampling": 0,
            "epochs": 1,
        },
        "thread_counts": thread_counts,
        "models": models,
        "objectives": objectives,
        "update_strategy": args.update_strategy,
        "corpus": str(corpus.relative_to(PACKAGE)),
        "corpus_sha256": corpus_hash,
        "config_sha256": config_hash,
        "stat_samples": observed_samples,
        "stack_profiles": profile_count,
        "training_metric_samples": len(metrics),
        "sample_warning": "some workloads completed in under 0.5 seconds; use --scale explicitly for longer samples"
        if duration_values and min(duration_values) < 0.5
        else None,
        "affinity_order": cpus,
        "cpu_topology": topology,
        "build": {
            "id": args.build_id,
            "c_flags": "-O3 -march=native -g -fno-omit-frame-pointer -pthread",
            "rust_profile": "opt-level=3,debug=2,lto=thin,codegen-units=1",
            "rust_flags": "-C target-cpu=native -C force-frame-pointers=yes",
        },
        "environment": {
            "platform": platform.platform(),
            "uname": platform.uname()._asdict(),
            "wsl2": "microsoft" in platform.release().lower(),
            "perf": version(["perf", "--version"]),
            "cc": version(["cc", "--version"]),
            "rustc": version(["rustc", "--version"]),
            "cargo": version(["cargo", "--version"]),
        },
        "git": {
            "head": version(["git", "rev-parse", "HEAD"]),
            "status_porcelain": run(
                ["git", "status", "--porcelain"], check=False
            ).stdout.splitlines(),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"complete: {output_dir} ({observed_samples} stat samples, {profile_count} stack profiles)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"profile failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
