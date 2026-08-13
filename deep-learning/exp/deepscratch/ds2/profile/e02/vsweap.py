"""Sweep vocabulary size for the implemented e02 Word2Vec update paths."""

from __future__ import annotations

from dataclasses import asdict
import gc
import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import save_figure
from exp.framework.plotting.theme import ACCENT_COLORS, MUTED
from exp.deepscratch.ds2.profile.paths import profile_cache
from mlprosection import Tensor
from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.nn.model.architecture import (
    CBOW,
    CBOWBatchAdapter,
    FusedNegativeSamplingCBOW,
    FusedNegativeSamplingSkipGram,
    SkipGram,
    SkipGramBatchAdapter,
)
from mlprosection.nn.objective import (
    FusedNegativeSampling,
    NegativeSampling,
    SoftmaxWithLoss,
)
from mlprosection.nn.sampling import UnigramSampler
from mlprosection.optim.SGD import Adam
from mlprosection.profiling import BenchmarkRunner

from .update import (
    _metadata,
    run_fused_update,
    run_implemented_update,
)


DEFAULT_RESULTS = profile_cache("e02")
DEFAULT_VOCAB_SIZES = (
    1_000,
    2_000,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
)
DEFAULT_CPU_VOCAB_SIZES = (
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
)
CONDITIONS = (
    "implemented-cbow-fs",
    "implemented-cbow-ns",
    "implemented-cbow-fused-ns",
    "implemented-skipgram-fs",
    "implemented-skipgram-ns",
    "implemented-skipgram-fused-ns",
)
EMBEDDING_SIZE = 100
CONTEXT_WIDTH = 10
NEGATIVE_SAMPLES = 5
_T_975 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


class SweepWorkload:
    """One current-implementation Word2Vec condition at a synthetic vocab size."""

    def __init__(
        self,
        condition: str,
        *,
        vocab_size: int,
        contexts: np.ndarray,
        targets: np.ndarray,
        backend,
    ) -> None:
        _, model_token, *variant_tokens = condition.split("-")
        variant = "-".join(variant_tokens)
        model_name = "CBOW" if model_token == "cbow" else "SkipGram"
        objective_name = (
            "FusedNegativeSampling"
            if variant == "fused-ns"
            else "NegativeSampling"
            if variant == "ns"
            else "FullSoftmax"
        )
        model_class = (
            (
                FusedNegativeSamplingCBOW
                if model_name == "CBOW"
                else FusedNegativeSamplingSkipGram
            )
            if objective_name == "FusedNegativeSampling"
            else CBOW
            if model_name == "CBOW"
            else SkipGram
        )
        self.fused = objective_name == "FusedNegativeSampling"
        self.backend = backend
        self.contexts = Tensor(
            backend.asarray(contexts, dtype=backend.xp.int64),
            backend=backend,
        )
        self.targets = Tensor(
            backend.asarray(targets, dtype=backend.xp.int64),
            backend=backend,
        )
        self.model = model_class(vocab_size, EMBEDDING_SIZE, backend=backend)
        self.adapter = (
            CBOWBatchAdapter()
            if model_name == "CBOW"
            else SkipGramBatchAdapter()
        )
        if objective_name in {"NegativeSampling", "FusedNegativeSampling"}:
            sampler = UnigramSampler.uniform(
                vocab_size,
                backend=backend,
                algorithm=UnigramSampler.CONDITIONAL_CDF,
            )
            objective_type = (
                FusedNegativeSampling
                if self.fused
                else NegativeSampling
            )
            self.objective = objective_type(
                vocab_size,
                negative_samples=NEGATIVE_SAMPLES,
                reduction="mean",
                sampler=sampler,
                backend=backend,
            )
        else:
            self.objective = SoftmaxWithLoss(
                reduction="mean",
                grouped_targets=model_name == "SkipGram",
                backend=backend,
            )
        params = [
            *(
                (f"model.{name}", parameter)
                for name, parameter in self.model.named_parameters()
            ),
            *(
                (f"objective.{name}", parameter)
                for name, parameter in self.objective.named_parameters()
            ),
        ]
        self.optimizer = Adam(params, lr=0.001)

    def update(self, batch_index: int, batch_size: int) -> None:
        start = batch_index * batch_size
        batch_x = self.contexts[start : start + batch_size]
        batch_t = self.targets[start : start + batch_size]
        update = run_fused_update if self.fused else run_implemented_update
        update(
            model=self.model,
            adapter=self.adapter,
            objective=self.objective,
            optimizer=self.optimizer,
            batch_x=batch_x,
            batch_t=batch_t,
        )


def run(
    *,
    devices: tuple[str, ...] = ("cuda:0",),
    vocab_sizes: tuple[int, ...] | None = None,
    conditions: tuple[str, ...] | None = None,
    batch_size: int = 100,
    warmup_updates: int = 20,
    measured_updates: int = 50,
    repetitions: int = 5,
    timing_source: str = "window",
    reverse_vocab_order: bool = False,
    output_dir: Path = DEFAULT_RESULTS,
) -> None:
    """Measure synchronized update distributions and windows at every sweep point."""
    if timing_source not in {"window", "event"}:
        raise ValueError("timing source must be 'window' or 'event'")
    selected_conditions = CONDITIONS if conditions is None else conditions
    unknown = set(selected_conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(
            "vocabulary sweep supports implemented conditions only: "
            f"{sorted(unknown)}"
        )
    if not selected_conditions:
        raise ValueError("vocabulary sweep requires at least one condition")
    if min(batch_size, measured_updates, repetitions) < 1 or warmup_updates < 0:
        raise ValueError(
            "batch size, measured updates, and repetitions must be positive; "
            "warmup updates must be non-negative"
        )
    if vocab_sizes is not None:
        _validate_vocab_sizes(vocab_sizes)

    for device in devices:
        device_vocab_sizes = (
            _default_vocab_sizes(device) if vocab_sizes is None else vocab_sizes
        )
        if reverse_vocab_order:
            device_vocab_sizes = tuple(reversed(device_vocab_sizes))
        backend = make_backend(
            BackendConfig(
                device=device,
                dtype="float32",
                seed=1,
                profile=device.startswith("cuda:"),
            )
        )
        rows: list[dict[str, object]] = []
        for vocab_size in device_vocab_sizes:
            contexts, targets = _synthetic_batches(
                vocab_size,
                batch_size=batch_size,
                update_count=(
                    1
                    + warmup_updates
                    + measured_updates
                    + measured_updates * repetitions
                ),
            )
            for condition in selected_conditions:
                backend.seed(1)
                np.random.seed(1)
                row = _measure_condition(
                    condition,
                    vocab_size=vocab_size,
                    contexts=contexts,
                    targets=targets,
                    backend=backend,
                    batch_size=batch_size,
                    warmup_updates=warmup_updates,
                    measured_updates=measured_updates,
                    repetitions=repetitions,
                )
                rows.append(row)
                if row["status"] == "ok":
                    print(
                        f"[{device}] V={vocab_size:,} {condition}: "
                        f"{float(row['update_ms']):.3f} ms/update",
                        flush=True,
                    )
                else:
                    print(
                        f"[{device}] V={vocab_size:,} {condition}: "
                        f"{row['status']} ({row['error']})",
                        flush=True,
                    )

        device_dir = output_dir / device.replace(":", "")
        device_dir.mkdir(parents=True, exist_ok=True)
        output = device_dir / "vsweap.json"
        payload = {
            "schema_version": 3,
            "metadata": {
                **_metadata(backend, stage="vsweap"),
                "method": (
                    "synthetic uniform vocabulary; current implemented update "
                    "path including dense Adam and post-update loss; device "
                    "synchronization at cold, warmup, event-distribution, and "
                    "repeated throughput-window boundaries"
                ),
                "embedding_size": EMBEDDING_SIZE,
                "context_width": CONTEXT_WIDTH,
                "negative_samples": NEGATIVE_SAMPLES,
                "batch_size": batch_size,
                "warmup_updates": warmup_updates,
                "measured_updates": measured_updates,
                "repetitions": repetitions,
                "vocab_sizes": list(device_vocab_sizes),
                "vocab_order": "descending" if reverse_vocab_order else "ascending",
                "timing_source": timing_source,
            },
            "results": rows,
            "crossovers": _crossovers(rows),
        }
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved: {output}", flush=True)
        figure = render_sweep(payload)
        figure_output = save_figure(figure, device_dir / "vsweap.png")
        plt.close(figure)
        print(f"saved: {figure_output}", flush=True)
        for model, figure in render_individual_sweeps(payload):
            individual_output = save_figure(
                figure,
                device_dir / f"vsweap-{model.lower()}.png",
            )
            plt.close(figure)
            print(f"saved: {individual_output}", flush=True)
        print(_render_crossovers(device, payload["crossovers"]), flush=True)


def _default_vocab_sizes(device: str) -> tuple[int, ...]:
    return (
        DEFAULT_CPU_VOCAB_SIZES
        if not device.startswith("cuda:")
        else DEFAULT_VOCAB_SIZES
    )


_PLOT_STYLES = {
    "implemented-cbow-fs": ("Full softmax", ACCENT_COLORS[0], "o", "-"),
    "implemented-cbow-ns": (
        "Negative sampling",
        ACCENT_COLORS[1],
        "s",
        "--",
    ),
    "implemented-cbow-fused-ns": (
        "Fused negative sampling",
        ACCENT_COLORS[3],
        "^",
        "-.",
    ),
    "implemented-skipgram-fs": (
        "Full softmax",
        ACCENT_COLORS[0],
        "o",
        "-",
    ),
    "implemented-skipgram-ns": (
        "Negative sampling",
        ACCENT_COLORS[1],
        "s",
        "--",
    ),
    "implemented-skipgram-fused-ns": (
        "Fused negative sampling",
        ACCENT_COLORS[3],
        "^",
        "-.",
    ),
}


def render_sweep(payload: dict[str, object]):
    """Render one repository-themed vocabulary/runtime figure per device."""
    rows = payload.get("results")
    metadata = payload.get("metadata")
    if not isinstance(rows, list) or not isinstance(metadata, dict):
        raise ValueError("invalid vocabulary sweep payload")
    selected = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "ok"
        and row.get("condition") in _PLOT_STYLES
    ]
    models = [
        model
        for model in ("CBOW", "SkipGram")
        if any(row.get("model") == model for row in selected)
    ]
    if not models:
        raise ValueError("vocabulary sweep has no plottable results")

    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.4 if len(models) == 1 else 10.0, 4.8),
        squeeze=False,
    )
    device = str(metadata.get("device", "unknown device"))
    device_label = (
        "GPU"
        if device.startswith("cuda:")
        else "CPU"
        if device.startswith("cpu")
        else device
    )
    for axis, model in zip(axes[0], models, strict=True):
        _plot_model(
            axis,
            selected,
            model,
            device_label,
            timing_source=str(metadata.get("timing_source", "window")),
            title=True,
        )
    return figure


def render_individual_sweeps(
    payload: dict[str, object],
) -> list[tuple[str, plt.Figure]]:
    """Render one untitled figure for each model represented in the payload."""
    rows = payload.get("results")
    metadata = payload.get("metadata")
    if not isinstance(rows, list) or not isinstance(metadata, dict):
        raise ValueError("invalid vocabulary sweep payload")
    selected = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "ok"
        and row.get("condition") in _PLOT_STYLES
    ]
    models = [
        model
        for model in ("CBOW", "SkipGram")
        if any(row.get("model") == model for row in selected)
    ]
    if not models:
        raise ValueError("vocabulary sweep has no plottable results")
    device = str(metadata.get("device", "unknown device"))
    device_label = (
        "GPU"
        if device.startswith("cuda:")
        else "CPU"
        if device.startswith("cpu")
        else device
    )
    figures = []
    for model in models:
        figure, axis = plt.subplots(figsize=(6.4, 4.8))
        _plot_model(
            axis,
            selected,
            model,
            device_label,
            timing_source=str(metadata.get("timing_source", "window")),
            title=False,
        )
        figures.append((model, figure))
    return figures


def _plot_model(
    axis,
    selected,
    model: str,
    device_label: str,
    *,
    timing_source: str,
    title: bool,
) -> None:
    if timing_source not in {"window", "event"}:
        raise ValueError("timing source must be 'window' or 'event'")
    model_rows = [row for row in selected if row.get("model") == model]
    conditions = [
        condition
        for condition in _PLOT_STYLES
        if any(row.get("condition") == condition for row in model_rows)
    ]
    for condition in conditions:
        label, color, marker, linestyle = _PLOT_STYLES[condition]
        condition_rows = sorted(
            (row for row in model_rows if row.get("condition") == condition),
            key=lambda row: int(row["vocab_size"]),
        )
        vocabulary = np.asarray(
            [int(row["vocab_size"]) for row in condition_rows], dtype=float
        )
        update_ms = np.asarray(
            [
                float(row["update_ms"])
                if timing_source == "window"
                else float(row["steady_event_timing"]["mean_ms"])
                for row in condition_rows
            ],
            dtype=float,
        )
        axis.plot(
            vocabulary,
            update_ms,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.6,
            markersize=5,
        )
        lower = [row.get("ci95_lower_ms") for row in condition_rows]
        upper = [row.get("ci95_upper_ms") for row in condition_rows]
        if timing_source == "window" and all(
            value is not None for value in (*lower, *upper)
        ):
            axis.fill_between(
                vocabulary,
                np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float),
                color=color,
                alpha=0.2,
                linewidth=0,
            )
    model_label = "Skip-gram" if model == "SkipGram" else model
    axis.set(
        xlabel="Vocabulary size",
        ylabel="Update time (ms)",
        xscale="log",
    )
    if title:
        axis.set_title(f"{model_label} · {device_label}")
    axis.grid(True, which="both", alpha=0.25, color=MUTED)
    axis.legend()


def _validate_vocab_sizes(vocab_sizes: tuple[int, ...]) -> None:
    if not vocab_sizes or min(vocab_sizes) < 2:
        raise ValueError("vocabulary sizes must contain integers of at least 2")
    if len(set(vocab_sizes)) != len(vocab_sizes):
        raise ValueError("vocabulary sizes must not contain duplicates")


def _measure_condition(
    condition: str,
    *,
    vocab_size: int,
    contexts: np.ndarray,
    targets: np.ndarray,
    backend,
    batch_size: int,
    warmup_updates: int,
    measured_updates: int,
    repetitions: int,
) -> dict[str, object]:
    model_name = "CBOW" if "-cbow-" in condition else "SkipGram"
    objective_name = (
        "FusedNegativeSampling"
        if condition.endswith("-fused-ns")
        else "NegativeSampling"
        if condition.endswith("-ns")
        else "FullSoftmax"
    )
    row: dict[str, object] = {
        "condition": condition,
        "implementation": "implemented",
        "model": model_name,
        "objective": objective_name,
        "vocab_size": vocab_size,
        "embedding_size": EMBEDDING_SIZE,
        "batch_size": batch_size,
        "warmup_updates": warmup_updates,
        "measured_updates": measured_updates,
        "repetitions": repetitions,
        "dense_parameter_optimizer_bytes": 8
        * vocab_size
        * EMBEDDING_SIZE
        * np.dtype(np.float32).itemsize,
    }
    workload = None
    next_index = 0
    try:
        workload = SweepWorkload(
            condition,
            vocab_size=vocab_size,
            contexts=contexts,
            targets=targets,
            backend=backend,
        )

        def update_once() -> None:
            nonlocal next_index
            workload.update(next_index, batch_size)
            next_index += 1

        result = BenchmarkRunner(backend).measure_update_protocol(
            f"vsweap.v{vocab_size}.{condition}",
            update_once,
            warmup_iterations=warmup_updates,
            measured_iterations=measured_updates,
            repetitions=repetitions,
        )
        interval = _mean_confidence_interval_95(result.timing)
        row.update(
            {
                "status": "ok",
                "update_ms": result.timing.mean_ms,
                "standard_error_ms": interval["standard_error_ms"],
                "ci95_lower_ms": interval["lower_ms"],
                "ci95_upper_ms": interval["upper_ms"],
                "ci95_half_width_ms": interval["half_width_ms"],
                "cold_ms": result.cold_ms,
                "warmup_total_ms": result.warmup_total_ms,
                "warmup_mean_ms": result.warmup_mean_ms,
                "steady_event_timing": asdict(result.event_timing),
                "timing": asdict(result.timing),
                "error": None,
            }
        )
    except Exception as exc:
        if not _is_out_of_memory(exc):
            raise
        row.update(
            {
                "status": "out_of_memory",
                "update_ms": None,
                "standard_error_ms": None,
                "ci95_lower_ms": None,
                "ci95_upper_ms": None,
                "ci95_half_width_ms": None,
                "cold_ms": None,
                "warmup_total_ms": None,
                "warmup_mean_ms": None,
                "steady_event_timing": None,
                "timing": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        workload = None
        gc.collect()
        _release_backend_memory(backend)
    return row


def _synthetic_batches(
    vocab_size: int,
    *,
    batch_size: int,
    update_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    sample_count = batch_size * update_count
    contexts = rng.integers(
        0,
        vocab_size,
        size=(sample_count, CONTEXT_WIDTH),
        dtype=np.int64,
    )
    targets = rng.integers(
        0,
        vocab_size,
        size=(sample_count, 1),
        dtype=np.int64,
    )
    return contexts, targets


def _mean_confidence_interval_95(timing) -> dict[str, float | None]:
    if timing.count < 2:
        return {
            "standard_error_ms": None,
            "lower_ms": None,
            "upper_ms": None,
            "half_width_ms": None,
        }
    standard_error = timing.stdev_ms / sqrt(timing.count)
    degrees_of_freedom = timing.count - 1
    critical_value = (
        _T_975[degrees_of_freedom]
        if degrees_of_freedom < len(_T_975)
        else 1.96
    )
    half_width = critical_value * standard_error
    return {
        "standard_error_ms": standard_error,
        "lower_ms": max(0.0, timing.mean_ms - half_width),
        "upper_ms": timing.mean_ms + half_width,
        "half_width_ms": half_width,
    }


def _crossovers(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result = {
        "CBOW": _crossover(rows, "CBOW", "NegativeSampling"),
        "CBOW-Fused": _crossover(rows, "CBOW", "FusedNegativeSampling"),
        "SkipGram": _crossover(rows, "SkipGram", "NegativeSampling"),
        "SkipGram-Fused": _crossover(
            rows,
            "SkipGram",
            "FusedNegativeSampling",
        ),
    }
    return {
        key: value
        for key, value in result.items()
        if value["comparisons"]
    }


def _crossover(
    rows: list[dict[str, object]],
    model: str,
    sampling_objective: str,
) -> dict[str, object]:
    by_key = {
        (int(row["vocab_size"]), str(row["objective"])): row
        for row in rows
        if row["model"] == model and row["status"] == "ok"
    }
    comparisons = []
    for vocab_size in sorted({key[0] for key in by_key}):
        ns = by_key.get((vocab_size, sampling_objective))
        fs = by_key.get((vocab_size, "FullSoftmax"))
        if ns is None or fs is None:
            continue
        ns_ms = float(ns["update_ms"])
        fs_ms = float(fs["update_ms"])
        ns_lower = ns["ci95_lower_ms"]
        ns_upper = ns["ci95_upper_ms"]
        fs_lower = fs["ci95_lower_ms"]
        fs_upper = fs["ci95_upper_ms"]
        if (
            ns_upper is not None
            and fs_lower is not None
            and float(ns_upper) < float(fs_lower)
        ):
            confidence_winner = sampling_objective
        elif (
            fs_upper is not None
            and ns_lower is not None
            and float(fs_upper) < float(ns_lower)
        ):
            confidence_winner = "FullSoftmax"
        else:
            confidence_winner = "Inconclusive"
        comparisons.append(
            {
                "vocab_size": vocab_size,
                "negative_sampling_ms": ns_ms,
                "full_softmax_ms": fs_ms,
                "negative_sampling_speedup": fs_ms / ns_ms,
                "point_estimate_winner": (
                    sampling_objective if ns_ms < fs_ms else "FullSoftmax"
                ),
                "confidence_winner": confidence_winner,
            }
        )
    first_observed = next(
        (
            comparison["vocab_size"]
            for comparison in comparisons
            if comparison["point_estimate_winner"] == sampling_objective
        ),
        None,
    )
    first_confirmed = next(
        (
            comparison["vocab_size"]
            for comparison, following in zip(comparisons, comparisons[1:])
            if comparison["confidence_winner"] == sampling_objective
            and following["confidence_winner"] == sampling_objective
        ),
        None,
    )
    return {
        "sampling_objective": sampling_objective,
        "first_observed_negative_sampling_win_vocab_size": first_observed,
        "first_confirmed_negative_sampling_win_vocab_size": first_confirmed,
        "confirmation_rule": (
            "non-overlapping 95% mean confidence intervals favor "
            f"{sampling_objective} at this and the next measured vocabulary size"
        ),
        "comparisons": comparisons,
    }


def _render_crossovers(
    device: str,
    crossovers: object,
) -> str:
    assert isinstance(crossovers, dict)
    lines = [f"\n# {device} vocabulary sweep crossover"]
    for model in ("CBOW", "CBOW-Fused", "SkipGram", "SkipGram-Fused"):
        summary = crossovers.get(model)
        if summary is None:
            continue
        assert isinstance(summary, dict)
        first = summary["first_confirmed_negative_sampling_win_vocab_size"]
        objective = str(summary["sampling_objective"])
        lines.append(
            f"- {model}: "
            + (
                f"no confirmed {objective} crossover"
                if first is None
                else f"confirmed {objective} crossover at V={int(first):,}"
            )
        )
    return "\n".join(lines)


def _is_out_of_memory(exc: Exception) -> bool:
    return isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError"


def _release_backend_memory(backend) -> None:
    if not backend.is_gpu:
        return
    backend.synchronize()
    backend.xp.get_default_memory_pool().free_all_blocks()
    backend.xp.get_default_pinned_memory_pool().free_all_blocks()
