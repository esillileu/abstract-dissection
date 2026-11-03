"""W2V1 paper-reproduction training orchestration."""

from __future__ import annotations

from typing import Any

from repro_core.context import RuntimePaths

from f2.suites.common.w2v.adapter import (
    run_w2v_with_coordinator,
)
from f2.suites.common.w2v.training import (
    W2VRunReceipt,
)


class W2V1Executor:
    """W2V1 study executor delegating lifecycle to F2 Coordinator."""

    suite_name = "w2v1"

    def execute_spec(
        self,
        config: dict[str, object],
        *,
        tracking_uri: str,
        paths: RuntimePaths,
        override: bool = False,
        progress_reporter: Any | None = None,
    ) -> W2VRunReceipt:
        return run_w2v_with_coordinator(
            config,
            suite_name=self.suite_name,
            tracking_uri=tracking_uri,
            paths=paths,
            override=override,
            progress_reporter=progress_reporter,
        )


EXECUTORS = {"word2vec": W2V1Executor()}


def get_executor(kind: str = "word2vec") -> W2V1Executor:
    try:
        return EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown W2V1 experiment kind: {kind}") from exc


__all__ = [
    "EXECUTORS",
    "W2V1Executor",
    "get_executor",
]
