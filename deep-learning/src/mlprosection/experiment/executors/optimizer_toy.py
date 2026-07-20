from __future__ import annotations

# ruff: noqa: E701, E702

import math
from pathlib import Path

from ..contracts import ExperimentResult
from ..executor import ExperimentContext
from ..registry import register_executor


@register_executor("optimizer_toy")
class OptimizerToyExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        optimizer = _mapping(config, "optimizer")
        x, y = -7.0, 2.0
        state = {name: 0.0 for name in ("vx", "vy", "hx", "hy", "mx", "my")}
        state["t"] = 0.0
        path_length = 0.0
        changes = {"x": 0, "y": 0}
        previous: tuple[float, float] | None = None
        history: list[tuple[str, int, str, float]] = []
        for step in range(int(_mapping(config, "training").get("max_updates", 30))):
            dx, dy = _update(optimizer, state, x / 10.0, 2.0 * y)
            x += dx; y += dy
            path_length += math.hypot(dx, dy)
            if previous is not None:
                changes["x"] += int((previous[0] < 0) != (dx < 0))
                changes["y"] += int((previous[1] < 0) != (dy < 0))
            previous = dx, dy
            metrics = {"opt/x": x, "opt/y": y, "opt/objective": x * x / 20 + y * y, "opt/distance_to_optimum": math.hypot(x, y), "opt/step_distance": math.hypot(dx, dy), "opt/cumulative_path_length": path_length, "opt/x_direction_changed": float(changes["x"]), "opt/y_direction_changed": float(changes["y"])}
            context.emit_metric(step, metrics)
            history.extend(("step", step, key, value) for key, value in metrics.items())
        return ExperimentResult(metrics={"final/status/success": 1.0, "final/status/nan_detected": 0.0, "final/status/inf_detected": 0.0, "final/status/diverged": 0.0, "final/system/total_updates": float(step + 1), "final/system/completed_epochs": 0.0, "final/system/samples_seen": 0.0, "final/opt/objective": x * x / 20 + y * y, "final/opt/distance_to_optimum": math.hypot(x, y), "final/opt/path_length": path_length, "final/opt/x_direction_changes": float(changes["x"]), "final/opt/y_direction_changes": float(changes["y"])}, artifact_root=_artifact_root(config), history=tuple(history))


def _update(config: dict[str, object], state: dict[str, float], gx: float, gy: float) -> tuple[float, float]:
    name, lr = str(config.get("name", "sgd")).lower(), float(config.get("learning_rate", 0.1))
    if name == "sgd": return -lr * gx, -lr * gy
    if name == "momentum":
        momentum = float(config.get("momentum", 0.9)); state["vx"] = momentum * state["vx"] - lr * gx; state["vy"] = momentum * state["vy"] - lr * gy; return state["vx"], state["vy"]
    if name == "adagrad":
        eps = float(config.get("eps", 1e-7)); state["hx"] += gx * gx; state["hy"] += gy * gy; return -lr * gx / (math.sqrt(state["hx"]) + eps), -lr * gy / (math.sqrt(state["hy"]) + eps)
    if name == "adam":
        beta1, beta2, eps = float(config.get("beta1", .9)), float(config.get("beta2", .999)), float(config.get("eps", 1e-7)); state["t"] += 1
        state["mx"] += (1 - beta1) * (gx - state["mx"]); state["my"] += (1 - beta1) * (gy - state["my"]); state["hx"] += (1 - beta2) * (gx * gx - state["hx"]); state["hy"] += (1 - beta2) * (gy * gy - state["hy"])
        lr_t = lr * math.sqrt(1 - beta2 ** state["t"]) / (1 - beta1 ** state["t"]); return -lr_t * state["mx"] / (math.sqrt(state["hx"]) + eps), -lr_t * state["my"] / (math.sqrt(state["hy"]) + eps)
    raise ValueError(f"unknown optimizer: {name}")


def _mapping(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    if not isinstance(value, dict): raise ValueError(f"{name} must be a mapping")
    return value


def _artifact_root(config: dict[str, object]) -> Path: return Path("experiments/results/runs") / str(config["atomic_run_id"])
