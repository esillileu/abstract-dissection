"""Sweep vocabulary size for the implemented e02 Word2Vec update paths."""

from __future__ import annotations

from dataclasses import asdict
import gc
import json
from math import sqrt
from pathlib import Path

import numpy as np

from mlprosection import Tensor
from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.nn.model.architecture import (
    CBOW,
    CBOWBatchAdapter,
    SkipGram,
    SkipGramBatchAdapter,
    SkipGramFullSoftmaxBatchAdapter,
)
from mlprosection.nn.objective import NegativeSampling, SoftmaxWithLoss
from mlprosection.nn.sampling import UnigramSampler
from mlprosection.optim.SGD import Adam
from mlprosection.profiling import BenchmarkRunner

from .update import ROOT, _metadata, run_implemented_update


DEFAULT_RESULTS = ROOT / "exp/ds2/profile/e02/results"
DEFAULT_VOCAB_SIZES = (
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
)
CONDITIONS = (
    "implemented-cbow-ns",
    "implemented-cbow-fs",
    "implemented-skipgram-ns",
    "implemented-skipgram-fs",
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
        _, model_token, objective_token = condition.split("-")
        model_name = "CBOW" if model_token == "cbow" else "SkipGram"
        objective_name = (
            "NegativeSampling" if objective_token == "ns" else "FullSoftmax"
        )
        model_class = CBOW if model_name == "CBOW" else SkipGram
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
            else (
                SkipGramBatchAdapter()
                if objective_name == "NegativeSampling"
                else SkipGramFullSoftmaxBatchAdapter()
            )
        )
        if objective_name == "NegativeSampling":
            sampler = UnigramSampler.uniform(
                vocab_size,
                backend=backend,
                algorithm=UnigramSampler.CONDITIONAL_CDF,
            )
            self.objective = NegativeSampling(
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
        run_implemented_update(
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
    vocab_sizes: tuple[int, ...] = DEFAULT_VOCAB_SIZES,
    conditions: tuple[str, ...] | None = None,
    batch_size: int = 100,
    warmup_updates: int = 20,
    measured_updates: int = 50,
    repetitions: int = 5,
    output_dir: Path = DEFAULT_RESULTS,
) -> None:
    """Measure synchronized update distributions and windows at every sweep point."""
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
    if not vocab_sizes or min(vocab_sizes) < 2:
        raise ValueError("vocabulary sizes must contain integers of at least 2")
    if len(set(vocab_sizes)) != len(vocab_sizes):
        raise ValueError("vocabulary sizes must not contain duplicates")

    for device in devices:
        backend = make_backend(
            BackendConfig(
                device=device,
                dtype="float32",
                seed=1,
                profile=device.startswith("cuda:"),
            )
        )
        rows: list[dict[str, object]] = []
        for vocab_size in vocab_sizes:
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
            "schema_version": 2,
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
                "vocab_sizes": list(vocab_sizes),
            },
            "results": rows,
            "crossovers": _crossovers(rows),
        }
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved: {output}", flush=True)
        print(_render_crossovers(device, payload["crossovers"]), flush=True)


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
    objective_name = "NegativeSampling" if condition.endswith("-ns") else "FullSoftmax"
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
    result: dict[str, dict[str, object]] = {}
    for model in ("CBOW", "SkipGram"):
        by_key = {
            (int(row["vocab_size"]), str(row["objective"])): row
            for row in rows
            if row["model"] == model and row["status"] == "ok"
        }
        comparisons = []
        for vocab_size in sorted({key[0] for key in by_key}):
            ns = by_key.get((vocab_size, "NegativeSampling"))
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
                confidence_winner = "NegativeSampling"
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
                        "NegativeSampling" if ns_ms < fs_ms else "FullSoftmax"
                    ),
                    "confidence_winner": confidence_winner,
                }
            )
        first_observed = next(
            (
                comparison["vocab_size"]
                for comparison in comparisons
                if comparison["point_estimate_winner"] == "NegativeSampling"
            ),
            None,
        )
        first_confirmed = next(
            (
                comparison["vocab_size"]
                for comparison, following in zip(comparisons, comparisons[1:])
                if comparison["confidence_winner"] == "NegativeSampling"
                and following["confidence_winner"] == "NegativeSampling"
            ),
            None,
        )
        result[model] = {
            "first_observed_negative_sampling_win_vocab_size": first_observed,
            "first_confirmed_negative_sampling_win_vocab_size": first_confirmed,
            "confirmation_rule": (
                "non-overlapping 95% mean confidence intervals favor "
                "NegativeSampling at this and the next measured vocabulary size"
            ),
            "comparisons": comparisons,
        }
    return result


def _render_crossovers(
    device: str,
    crossovers: object,
) -> str:
    assert isinstance(crossovers, dict)
    lines = [f"\n# {device} vocabulary sweep crossover"]
    for model in ("CBOW", "SkipGram"):
        summary = crossovers[model]
        assert isinstance(summary, dict)
        first = summary["first_confirmed_negative_sampling_win_vocab_size"]
        lines.append(
            f"- {model}: "
            + (
                "no confirmed NS crossover"
                if first is None
                else f"confirmed NS crossover at V={int(first):,}"
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
