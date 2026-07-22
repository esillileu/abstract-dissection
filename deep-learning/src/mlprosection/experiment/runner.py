from __future__ import annotations

from importlib import import_module

from .executor import ExperimentContext
from .registry import get_executor


def run_config(
    config: dict[str, object],
    context: ExperimentContext | None = None,
    *,
    executor_module: str | None = None,
):
    """Run through an optional explicitly selected experiment-domain adapter.

    ``src`` owns only the registry contract. Domain CLIs provide their module;
    callers that register an executor themselves may omit it.
    """
    module = import_module(executor_module) if executor_module is not None else None
    context = context or ExperimentContext()
    if str(config["kind"]) == "observation" and module is not None:
        resolver = getattr(module, "get_observation_executor", None)
        if callable(resolver):
            return resolver(config).run(config, context)
    return get_executor(str(config["kind"])).run(config, context)
