from __future__ import annotations

from .observation_activation import ActivationObservationExecutor
from .observation_gradient import GradientCheckObservationExecutor
from .observation_trajectory import OptimizerTrajectoryObservationExecutor


def get_observation_executor(config: dict[str, object]):
    group_id = str(config.get("execution_group_id", ""))
    if group_id == "GO01":
        return OptimizerTrajectoryObservationExecutor()
    if group_id == "GO02":
        return ActivationObservationExecutor()
    if group_id == "GO03":
        return GradientCheckObservationExecutor()
    raise ValueError(f"unknown DS1 observation group: {group_id}")
