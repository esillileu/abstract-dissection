from __future__ import annotations

# ruff: noqa: E701, E702

from pathlib import Path

import numpy as np

from ..contracts import ExperimentResult
from ..executor import ExperimentContext
from ..registry import register_executor


@register_executor("activation_probe")
class ActivationProbeExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        model, initializer = _mapping(config, "model"), _mapping(config, "initializer")
        width, depth, samples = int(model.get("width", 100)), int(model.get("depth", 5)), int(_mapping(config, "dataset").get("train_size", 1000))
        activation, name = str(model.get("activation", "relu")), str(initializer.get("name", "he"))
        rng = np.random.default_rng(int(config.get("seed", 0)))
        values = rng.standard_normal((samples, width), dtype=np.float32)
        history: list[tuple[str, int, str, float]] = []; final: dict[str, float] = {"final/status/success": 1.0, "final/status/nan_detected": 0.0, "final/status/inf_detected": 0.0, "final/status/diverged": 0.0, "final/system/total_updates": 0.0, "final/system/completed_epochs": 0.0, "final/system/samples_seen": float(samples)}
        first_std = 0.0; max_saturation = max_zero = mean_shift = 0.0
        for index in range(depth):
            std = _std(name, width, initializer); weights = rng.standard_normal((width, width), dtype=np.float32) * std
            values = _activate(activation, values @ weights)
            stats = _stats(values, activation)
            first_std = stats["std"] if index == 0 else first_std; max_saturation = max(max_saturation, stats["saturation_ratio"]); max_zero = max(max_zero, stats["zero_ratio"]); mean_shift += abs(stats["mean"])
            for metric, value in stats.items():
                key = f"layer/{index + 1:02d}/activation/{metric}"; final[key] = value; history.append(("step", index, key, value))
            context.emit_metric(index, stats)
        final.update({"final/activation/std_retention_ratio": final[f"layer/{depth:02d}/activation/std"] / first_std if first_std else 0.0, "final/activation/mean_absolute_shift": mean_shift / depth, "final/activation/max_saturation_ratio": max_saturation, "final/activation/max_zero_ratio": max_zero})
        return ExperimentResult(metrics=final, artifact_root=Path("experiments/results/runs") / str(config["atomic_run_id"]), history=tuple(history))


def _mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if not isinstance(value, dict): raise ValueError(f"{key} must be a mapping")
    return value


def _std(name: str, width: int, config: dict[str, object]) -> float:
    if name == "xavier": return (1 / width) ** .5
    if name == "he": return (2 / width) ** .5
    return float(config.get("scale", 1.0))


def _activate(name: str, values: np.ndarray) -> np.ndarray:
    if name == "relu": return np.maximum(values, 0)
    if name == "sigmoid": return 1 / (1 + np.exp(-values))
    if name == "tanh": return np.tanh(values)
    raise ValueError(f"unknown activation: {name}")


def _stats(values: np.ndarray, activation: str) -> dict[str, float]:
    saturation = float(((values < .01) | (values > .99)).mean()) if activation == "sigmoid" else float((np.abs(values) > .99).mean()) if activation == "tanh" else 0.0
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max()), "p01": float(np.percentile(values, 1)), "p25": float(np.percentile(values, 25)), "median": float(np.percentile(values, 50)), "p75": float(np.percentile(values, 75)), "p99": float(np.percentile(values, 99)), "zero_ratio": float((values == 0).mean()), "saturation_ratio": saturation, "nonfinite_ratio": float((~np.isfinite(values)).mean())}
