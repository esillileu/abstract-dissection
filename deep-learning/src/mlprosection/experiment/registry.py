from __future__ import annotations

from .executor import ExperimentExecutor

_EXECUTORS: dict[str, ExperimentExecutor] = {}


def register_executor(kind: str):
    def decorator(executor: ExperimentExecutor) -> ExperimentExecutor:
        if kind in _EXECUTORS:
            raise ValueError(f"executor already registered: {kind}")
        _EXECUTORS[kind] = executor() if isinstance(executor, type) else executor
        return executor
    return decorator


def get_executor(kind: str) -> ExperimentExecutor:
    try:
        return _EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown experiment kind: {kind}") from exc
