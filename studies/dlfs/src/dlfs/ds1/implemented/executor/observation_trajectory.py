from __future__ import annotations

import numpy as np

from repro_core.context import ExperimentContext
from repro_core.context.contracts import ExperimentResult

from ..records import DS1Records
from .common import _artifact_root, _mapping, _write_rows


class OptimizerTrajectoryObservationExecutor:
    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        artifact_root = _artifact_root(context)
        observations = artifact_root / "observations"
        observations.mkdir(parents=True, exist_ok=True)
        optimizer = str(_mapping(config, "optimizer").get("name", "toy_sgd"))
        lr = float(_mapping(config, "optimizer").get("learning_rate", 0.95))
        max_updates = int(_mapping(config, "training").get("max_updates", 30))
        progress = context.metadata.get("progress_reporter")
        if progress is not None:
            progress.set_total_updates(max_updates)
        x, y = -7.0, 2.0
        vx = vy = gx2 = gy2 = mx = my = ux = uy = 0.0
        beta1, beta2, eps, momentum = 0.9, 0.999, 1e-7, 0.9
        rows = []
        for update in range(max_updates):
            objective = x * x / 20.0 + y * y
            grad_x, grad_y = x / 10.0, 2.0 * y
            rows.append(
                {
                    "update": update,
                    "x": x,
                    "y": y,
                    "objective": objective,
                    "grad_x": grad_x,
                    "grad_y": grad_y,
                }
            )
            name = optimizer.lower().removeprefix("toy_").replace("-", "_")
            if name == "momentum":
                vx = momentum * vx - lr * grad_x
                vy = momentum * vy - lr * grad_y
                x += vx
                y += vy
            elif name == "adagrad":
                gx2 += grad_x * grad_x
                gy2 += grad_y * grad_y
                x -= lr * grad_x / (np.sqrt(gx2) + eps)
                y -= lr * grad_y / (np.sqrt(gy2) + eps)
            elif name == "adam":
                step = update + 1
                mx = beta1 * mx + (1 - beta1) * grad_x
                my = beta1 * my + (1 - beta1) * grad_y
                ux = beta2 * ux + (1 - beta2) * grad_x * grad_x
                uy = beta2 * uy + (1 - beta2) * grad_y * grad_y
                x -= (
                    lr
                    * (mx / (1 - beta1**step))
                    / (np.sqrt(ux / (1 - beta2**step)) + eps)
                )
                y -= (
                    lr
                    * (my / (1 - beta1**step))
                    / (np.sqrt(uy / (1 - beta2**step)) + eps)
                )
            elif name == "sgd":
                x -= lr * grad_x
                y -= lr * grad_y
            else:
                raise ValueError(f"unknown toy optimizer: {optimizer}")
            if progress is not None:
                progress.advance_to(update + 1)
        _write_rows(
            observations / "trajectory.csv",
            rows,
            ["update", "x", "y", "objective", "grad_x", "grad_y"],
        )
        records = DS1Records()
        records.bind_artifact_root(artifact_root)
        records.flush()
        return ExperimentResult(
            metrics={
                "final/status/success": 1.0,
                "final/system/total_updates": float(max_updates),
                "final/system/completed_epochs": 0.0,
                "final/system/samples_seen": 0.0,
            },
            artifact_root=artifact_root,
            metric_rows=tuple(
                (int(row["update"]), f"update/trajectory/{key}", float(row[key]))
                for row in rows
                for key in ("x", "y", "objective", "grad_x", "grad_y")
            ),
        )
