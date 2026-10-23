"""W2V1 paper-reproduction training orchestration."""

from __future__ import annotations

from f2.suites.w2v.executor import (
    Word2VecExecutor,
    Word2VecResult,
    _corpus_digest,
    _mapping,
    create_session,
)

W2V1Result = Word2VecResult


class W2V1Executor(Word2VecExecutor):
    """W2V1 study executor specializing suite name to w2v1."""

    suite_name = "w2v1"


EXECUTORS = {"word2vec": W2V1Executor()}


def get_executor(kind: str) -> W2V1Executor:
    try:
        return EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown W2V1 experiment kind: {kind}") from exc


__all__ = [
    "EXECUTORS",
    "W2V1Executor",
    "W2V1Result",
    "_corpus_digest",
    "_mapping",
    "create_session",
    "get_executor",
]
