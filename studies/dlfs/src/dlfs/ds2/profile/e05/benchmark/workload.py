"""Full BetterRnnlm workload definition and iterative benchmark execution."""

from __future__ import annotations

import math
import time

import numpy as np
from deepscratch.core import Tensor
from deepscratch.datasets import load_ptb
from deepscratch.nn.model.architecture.recurrent import BetterRnnlm
from deepscratch.optim.SGD import SGD
from deepscratch.optim.transform import ClipGradNorm

from ..phase1 import replace_better_rnnlm_lstms as replace_phase1_lstms
from ..phase2 import replace_better_rnnlm_lstms as replace_phase2_lstms
from ..phase3 import (
    Phase3TemporalSoftmaxCrossEntropy,
    UnfusedTemporalSoftmaxCrossEntropy,
)
from ..reference import replace_better_rnnlm_lstms
from .timing import _stats

PHASES = ("model_forward", "objective", "backward", "clipping", "sgd", "state_detach")


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
        self.model = BetterRnnlm(self.vocab_size, 650, 650, 0.5, backend=backend)
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
        named = [
            (f"model.{name}", value) for name, value in self.model.named_parameters()
        ]
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
    workload = BetterRnnlmWorkload(
        backend, implementation=implementation, profile=profile
    )
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
        "cuda_phases": {
            name: _stats(values) for name, values in phase_samples.items() if values
        },
        "updates_per_epoch": updates,
        "estimated_epoch_seconds": mean_ms * updates / 1_000,
        "estimated_40_epoch_hours": mean_ms * updates * 40 / 3_600_000,
        "excluded_from_estimate": ["evaluation", "checkpoint", "MLflow I/O"],
        "last_loss": last_loss,
    }
