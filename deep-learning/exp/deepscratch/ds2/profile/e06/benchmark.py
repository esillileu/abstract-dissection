"""Measure isolated e06 Seq2seq updates without MLflow or checkpoints."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from mlprosection import Tensor
from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.datasets import load_sequence
from mlprosection.nn.model.architecture import PeekySeq2seq, Seq2seq
from mlprosection.nn.objective import TemporalSoftmaxCrossEntropy
from mlprosection.optim.SGD import Adam
from mlprosection.optim.transform import ClipGradNorm
from exp.deepscratch.ds2.profile.paths import profile_cache


DEFAULT_RESULTS = profile_cache("e06")
CONDITIONS = {
    "vanilla-forward": (Seq2seq, False),
    "vanilla-reverse": (Seq2seq, True),
    "peeky-forward": (PeekySeq2seq, False),
    "peeky-reverse": (PeekySeq2seq, True),
}


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _sync(backend) -> None:
    if backend.is_gpu:
        backend.synchronize()


def _timed(backend, operation) -> float:
    _sync(backend)
    start = time.perf_counter()
    operation()
    _sync(backend)
    return (time.perf_counter() - start) * 1_000


def _environment(backend) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": backend.name,
        "device": backend.device,
        "dtype": backend.dtype_name,
        "cupy": getattr(backend.xp, "__version__", None),
        "git_commit": _git_commit(),
    }


def _stats(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "stdev_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "samples_ms": [float(value) for value in values],
    }


class Workload:
    def __init__(self, backend, *, model_class, reverse: bool) -> None:
        data = load_sequence(
            "addition.txt", seed=1984, split_algorithm="legacy_numpy_randomstate"
        )
        x_train, t_train = data["train"]
        if reverse:
            x_train = x_train[:, ::-1]
        xp = backend.xp
        self.backend = backend
        self.x = Tensor(xp.asarray(x_train, dtype=xp.int64), backend=backend)
        self.t = Tensor(xp.asarray(t_train, dtype=xp.int64), backend=backend)
        self.position = 0
        self.batch_size = 128
        self.model = model_class(
            len(data["char_to_id"]), 16, 128, backend=backend
        )
        self.objective = TemporalSoftmaxCrossEntropy(reduction="mean", backend=backend)
        parameters = [
            *((f"model.{name}", value) for name, value in self.model.named_parameters()),
            *((f"objective.{name}", value) for name, value in self.objective.named_parameters()),
        ]
        self.optimizer = Adam(
            parameters, lr=0.001, pre_step_hooks=[ClipGradNorm(5.0)]
        )

    def update(self) -> float:
        start = self.position
        end = start + self.batch_size
        if end > len(self.x):
            self.position = 0
            start, end = 0, self.batch_size
        batch_x, batch_t = self.x[start:end], self.t[start:end]
        self.position = end
        decoder_x, objective_t = batch_t[:, :-1], batch_t[:, 1:]
        prediction = self.model.forward(batch_x, decoder_x)
        result = self.objective.forward(prediction, objective_t)
        self.model.backward(self.objective.backward())
        self.optimizer.update()
        return self.backend.scalar_to_float(result.loss.data)


def run(
    *,
    device: str = "cuda:0",
    warmup: int = 20,
    iterations: int = 100,
    repetitions: int = 5,
    output_dir: Path = DEFAULT_RESULTS,
    conditions: tuple[str, ...] = tuple(CONDITIONS),
) -> Path:
    backend = make_backend(BackendConfig(device=device, dtype="float32", seed=1))
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown e06 profiling condition(s): {', '.join(unknown)}")
    selected_conditions = tuple(CONDITIONS) if not conditions else conditions
    result: dict[str, object] = {
        "schema_version": 1,
        "environment": _environment(backend),
        "protocol": {
            "dataset": "addition.txt",
            "split_seed": 1984,
            "batch_size": 128,
            "drop_last": True,
            "warmup": warmup,
            "iterations": iterations,
            "repetitions": repetitions,
            "excluded": ["evaluation", "checkpoint", "MLflow I/O"],
        },
        "conditions": {},
    }
    for name in selected_conditions:
        model_class, reverse = CONDITIONS[name]
        workload = Workload(backend, model_class=model_class, reverse=reverse)
        for _ in range(warmup):
            workload.update()
        samples = []
        losses = []
        for _ in range(repetitions):
            window = []
            for _ in range(iterations):
                elapsed = _timed(backend, lambda: losses.append(workload.update()))
                window.append(elapsed)
            samples.append(float(np.mean(window)))
        result["conditions"][name] = {
            "reverse": reverse,
            "model": model_class.__name__,
            "parameter_count": sum(parameter.data.size for _, parameter in workload.model.named_parameters()),
            "update": _stats(samples),
            "last_loss": losses[-1],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "benchmark.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return path


if __name__ == "__main__":
    run()
