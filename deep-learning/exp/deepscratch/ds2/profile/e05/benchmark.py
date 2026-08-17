"""CUDA-event and wall-clock benchmarks for BetterRnnlm Phase 1."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from mlprosection import Tensor
from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.datasets import load_ptb
from mlprosection.nn.layers import TimeLSTM
from mlprosection.nn.model.architecture.recurrent import BetterRnnlm
from mlprosection.optim.SGD import SGD
from mlprosection.optim.transform import ClipGradNorm
from exp.deepscratch.ds2.profile.paths import profile_measurements

from .reference import ReferenceTimeLSTM, replace_better_rnnlm_lstms
from .phase1 import Phase1TimeLSTM, replace_better_rnnlm_lstms as replace_phase1_lstms
from .phase2 import Phase2TimeLSTM, replace_better_rnnlm_lstms as replace_phase2_lstms
from .phase3 import (
    Phase3TemporalSoftmaxCrossEntropy,
    UnfusedTemporalSoftmaxCrossEntropy,
)


DEFAULT_RESULTS = profile_measurements("e05")
PHASES = ("model_forward", "objective", "backward", "clipping", "sgd", "state_detach")


def _stats(samples: list[float]) -> dict[str, float | list[float]]:
    flat = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(flat.mean()),
        "stdev_ms": float(flat.std(ddof=1)) if len(flat) > 1 else 0.0,
        "p50_ms": float(np.percentile(flat, 50)),
        "p95_ms": float(np.percentile(flat, 95)),
        "samples_ms": [float(value) for value in samples],
    }


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def environment(backend) -> dict[str, object]:
    values: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": backend.name,
        "device": backend.device,
        "dtype": backend.dtype_name,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
    }
    if backend.is_gpu:
        cp = backend.xp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]
        values.update(
            {
                "cupy": cp.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver": cp.cuda.runtime.driverGetVersion(),
                "gpu": name.decode() if isinstance(name, bytes) else str(name),
                "nvidia_smi": _command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem",
                        "--format=csv,noheader",
                    ]
                ),
            }
        )
    return values


def _event_ms(backend, operation: Callable[[], None]) -> float:
    if not backend.is_gpu:
        start = time.perf_counter()
        operation()
        return (time.perf_counter() - start) * 1_000
    start = backend.xp.cuda.Event()
    stop = backend.xp.cuda.Event()
    start.record()
    operation()
    stop.record()
    stop.synchronize()
    return float(backend.xp.cuda.get_elapsed_time(start, stop))


def _repeat(
    operation: Callable[[], None], backend, *, warmup: int, iterations: int, repetitions: int
) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    backend.synchronize()
    windows = []
    for _ in range(repetitions):
        elapsed = _event_ms(backend, lambda: [operation() for _ in range(iterations)])
        windows.append(elapsed / iterations)
    return _stats(windows)


def benchmark_timelstm(
    backend,
    *,
    implementation: str,
    warmup: int,
    iterations: int,
    repetitions: int,
    shape: tuple[int, int, int, int] = (20, 35, 650, 650),
) -> dict[str, object]:
    n, time_size, input_size, hidden_size = shape
    xp = backend.xp
    cls = {
        "reference": ReferenceTimeLSTM,
        "production": TimeLSTM,
        "phase1": Phase1TimeLSTM,
        "phase2": Phase2TimeLSTM,
        "phase3": Phase2TimeLSTM,
    }[implementation]
    xp.random.seed(20260811)
    layer = cls(input_size, hidden_size, stateful=True, backend=backend)
    xs = Tensor(xp.random.randn(n, time_size, input_size).astype(xp.float32), backend=backend)
    dhs = Tensor(xp.random.randn(n, time_size, hidden_size).astype(xp.float32), backend=backend)
    h0 = xp.random.randn(n, hidden_size).astype(xp.float32)
    c0 = xp.random.randn(n, hidden_size).astype(xp.float32)

    def forward() -> None:
        layer.set_state(h0, c0)
        layer.forward(xs)

    forward_result = _repeat(
        forward, backend, warmup=warmup, iterations=iterations, repetitions=repetitions
    )
    layer.set_state(h0, c0)
    layer.forward(xs)
    backward_result = _repeat(
        lambda: layer.backward(dhs),
        backend,
        warmup=warmup,
        iterations=iterations,
        repetitions=repetitions,
    )
    return {
        "shape": {"N": n, "T": time_size, "D": input_size, "H": hidden_size},
        "forward": forward_result,
        "backward": backward_result,
    }


class BetterRnnlmWorkload:
    def __init__(self, backend, *, implementation: str, profile: bool = False) -> None:
        self.backend = backend
        self.xp = backend.xp
        self.xp.random.seed(20260811)
        ptb = load_ptb()
        self.corpus = self.xp.asarray(ptb["train"], dtype=self.xp.int64)
        self.vocab_size = len(ptb["word_to_id"])
        if self.vocab_size != 10_000:
            raise RuntimeError(f"expected PTB vocabulary 10000, got {self.vocab_size}")
        self.model = BetterRnnlm(
            self.vocab_size, 650, 650, 0.5, backend=backend
        )
        if implementation == "reference":
            replace_better_rnnlm_lstms(self.model)
        elif implementation == "phase1":
            replace_phase1_lstms(self.model)
        elif implementation in {"phase2", "phase3"}:
            replace_phase2_lstms(self.model)
        objective_cls = (
            Phase3TemporalSoftmaxCrossEntropy
            if implementation == "phase3"
            else UnfusedTemporalSoftmaxCrossEntropy
        )
        self.objective = objective_cls(reduction="mean", backend=backend)
        named = [(f"model.{name}", value) for name, value in self.model.named_parameters()]
        self.optimizer = SGD(named, lr=20.0)
        self.clipper = ClipGradNorm(0.25)
        self.batch_size, self.time_size, self.time_index = 20, 35, 0
        jump = (len(self.corpus) - 1) // self.batch_size
        self.batch_offsets = self.xp.arange(self.batch_size) * jump
        self.time_offsets = self.xp.arange(self.time_size)
        self.profile = profile

    @property
    def updates_per_epoch(self) -> int:
        return (len(self.corpus) - 1) // (self.batch_size * self.time_size)

    def batch(self) -> tuple[Tensor, Tensor]:
        size = len(self.corpus) - 1
        positions = (
            self.batch_offsets[:, None] + self.time_index + self.time_offsets[None, :]
        ) % size
        self.time_index = (self.time_index + self.time_size) % size
        return (
            Tensor(self.corpus[positions], backend=self.backend),
            Tensor(self.corpus[positions + 1], backend=self.backend),
        )

    def update(self, *, record_events: bool = False):
        events = []

        def mark(name: str) -> None:
            if record_events and self.backend.is_gpu:
                event = self.xp.cuda.Event()
                event.record()
                events.append((name, event))

        start_wall = time.perf_counter()
        xs, targets = self.batch()
        # Batch generation belongs to the authoritative wall window but is not
        # folded into the model-forward CUDA phase.
        mark("start")
        with self.backend.range("e05/full_update/model_forward"):
            prediction = self.model.forward(xs)
        mark("model_forward")
        with self.backend.range("e05/full_update/objective"):
            result = self.objective.forward(prediction, targets)
        mark("objective")
        with self.backend.range("e05/full_update/backward"):
            self.model.backward(self.objective.backward())
        mark("backward")
        named = [(name, p) for name, p in self.optimizer.params if p.grad is not None]
        with self.backend.range("e05/full_update/clipping"):
            self.clipper(named)
        mark("clipping")
        with self.backend.range("e05/full_update/sgd"):
            for name, parameter in named:
                self.optimizer.update_one(name, parameter)
        mark("sgd")
        with self.backend.range("e05/full_update/state_detach"):
            self.model.detach_runtime_state()
            for layer in self.model.lstm_layers:
                layer.detach_state()
        mark("state_detach")
        return start_wall, result, events


def benchmark_full_update(
    backend,
    *,
    implementation: str,
    warmup: int,
    iterations: int,
    repetitions: int,
    profile: bool = False,
) -> dict[str, object]:
    workload = BetterRnnlmWorkload(backend, implementation=implementation, profile=profile)
    for _ in range(warmup):
        workload.update()
    backend.synchronize()
    window_samples: list[float] = []
    phase_samples = {name: [] for name in PHASES}
    last_loss = math.nan
    for _ in range(repetitions):
        phase_totals = {name: 0.0 for name in PHASES}
        window_start = time.perf_counter()
        records = []
        for _ in range(iterations):
            records.append(workload.update(record_events=backend.is_gpu))
        backend.synchronize()
        elapsed = (time.perf_counter() - window_start) * 1_000 / iterations
        window_samples.append(elapsed)
        if backend.is_gpu:
            for _wall, result, events in records:
                previous = events[0][1]
                for name, event in events[1:]:
                    phase_totals[name] += float(
                        backend.xp.cuda.get_elapsed_time(previous, event)
                    )
                    previous = event
                last_loss = backend.scalar_to_float(result.loss.data)
            for name in PHASES:
                phase_samples[name].append(phase_totals[name] / iterations)
        else:
            last_loss = backend.scalar_to_float(records[-1][1].loss.data)
    updates = workload.updates_per_epoch
    mean_ms = float(np.mean(window_samples))
    return {
        "window": _stats(window_samples),
        "cuda_phases": {name: _stats(values) for name, values in phase_samples.items() if values},
        "updates_per_epoch": updates,
        "estimated_epoch_seconds": mean_ms * updates / 1_000,
        "estimated_40_epoch_hours": mean_ms * updates * 40 / 3_600_000,
        "excluded_from_estimate": ["evaluation", "checkpoint", "MLflow I/O"],
        "last_loss": last_loss,
    }


def run(
    *,
    stage: str,
    device: str = "cuda:0",
    warmup: int = 20,
    iterations: int = 50,
    repetitions: int = 5,
    output_dir: Path = DEFAULT_RESULTS,
    timelstm_only: bool = False,
    profile: bool = False,
) -> Path:
    if stage not in {"baseline", "phase1", "phase2", "phase3"}:
        raise ValueError("stage must be baseline, phase1, phase2, or phase3")
    implementation = {
        "baseline": "reference",
        "phase1": "phase1",
        "phase2": "phase2",
        "phase3": "phase3",
    }[stage]
    backend = make_backend(
        BackendConfig(device=device, dtype="float32", seed=20260811, profile=profile)
    )
    result = {
        "schema_version": 1,
        "stage": stage,
        "implementation": implementation,
        "environment": environment(backend),
        "protocol": {"warmup": warmup, "iterations": iterations, "repetitions": repetitions},
        "timelstm": benchmark_timelstm(
            backend,
            implementation=implementation,
            warmup=warmup,
            iterations=iterations,
            repetitions=repetitions,
        ),
    }
    if not timelstm_only:
        result["full_update"] = benchmark_full_update(
            backend,
            implementation=implementation,
            warmup=warmup,
            iterations=iterations,
            repetitions=repetitions,
            profile=profile,
        )
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / "benchmark.json"
    if stage == "baseline" and path.exists():
        raise FileExistsError(f"refusing to overwrite immutable baseline: {path}")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return path
