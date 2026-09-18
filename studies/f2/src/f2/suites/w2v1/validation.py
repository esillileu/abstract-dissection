"""Validation and human-readable reporting for local W2V1 results."""

from __future__ import annotations

import json
from pathlib import Path

from f2.suites.w2v.artifacts import load_checkpoint, load_lookup_artifact
from repro_core.context import RuntimePaths

SLOT_ID = "w2v1-local-smoke-s1"


def _root(paths: RuntimePaths) -> Path:
    return paths.run_staging(
        domain="f2", suite="w2v1", study="table2", variant="local", run_key=SLOT_ID
    )


def check_latest(paths: RuntimePaths | None = None) -> str:
    paths = paths or RuntimePaths.from_environment()
    root = _root(paths)
    report_path = root / "result.json"
    if not report_path.is_file():
        raise ValueError("W2V1 result is missing; run the local-smoke variant first")
    report = json.loads(report_path.read_text())
    artifacts = report["artifacts"]
    checkpoint = root / artifacts["checkpoint"]
    lookup = root / artifacts["lookup"]
    metrics = root / artifacts["metrics"]
    load_checkpoint(checkpoint)
    load_lookup_artifact(lookup)
    if not metrics.is_file() or metrics.stat().st_size == 0:
        raise ValueError("W2V1 dense observations are missing")
    if report["evaluation"]["total_count"] < 1:
        raise ValueError("W2V1 evaluation contains no questions")
    return f"W2V1 local result complete: {report_path}"


def analyze_latest(paths: RuntimePaths | None = None) -> str:
    paths = paths or RuntimePaths.from_environment()
    check_latest(paths)
    report = json.loads((_root(paths) / "result.json").read_text())
    evaluation = report["evaluation"]
    output = paths.analysis_output("f2", "w2v1") / "summary.md"
    output.write_text(
        "# W2V1 local vertical slice\n\n"
        f"- Slot: `{SLOT_ID}`\n"
        f"- Epochs: {len(report['epochs'])}\n"
        f"- Analogy accuracy: {evaluation['score']:.6f}\n"
        f"- Coverage: {report['coverage']:.6f}\n"
        f"- Valid questions: {evaluation['valid_count']}/{evaluation['total_count']}\n"
    )
    return f"W2V1 analysis written: {output}"


__all__ = ["analyze_latest", "check_latest"]
