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
    """Run through an explicitly selected experiment-domain adapter.

    ``src`` owns only the registration contract.  The CLI supplies the domain
    module (for example ``exp.ds1.executor``); the fallback preserves the
    pre-domain public API for existing library callers.
    """
    if executor_module is None:
        from . import executors  # noqa: F401  # legacy built-in registration
    else:
        import_module(executor_module)
    context = context or ExperimentContext()
    return get_executor(str(config["kind"])).run(config, context)
