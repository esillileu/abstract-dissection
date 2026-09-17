"""Timing utilities, event measurements, and system profiling environment."""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Callable

import numpy as np


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
    operation: Callable[[], None],
    backend,
    *,
    warmup: int,
    iterations: int,
    repetitions: int,
) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    backend.synchronize()
    windows = []
    for _ in range(repetitions):
        elapsed = _event_ms(backend, lambda: [operation() for _ in range(iterations)])
        windows.append(elapsed / iterations)
    return _stats(windows)
