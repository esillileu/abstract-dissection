from __future__ import annotations

from .executor import ExperimentContext
from .registry import get_executor
from . import executors  # noqa: F401  # register built-in executor kinds


def run_config(config: dict[str, object], context: ExperimentContext | None = None):
    context = context or ExperimentContext()
    return get_executor(str(config["kind"])).run(config, context)
