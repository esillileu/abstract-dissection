from __future__ import annotations

import os
import re
from pathlib import Path


def test_retired_tracking_contracts_are_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    retired = tuple(
        re.compile(rf"(?<![A-Z0-9_]){re.escape(token)}(?![A-Z0-9_])")
        for token in (
            "MLFLOW_" + "F1_URL",
            "MLFLOW_" + "F2_URL",
            "MLFLOW_F1_" + "DATABASE_URL",
            "REPRO_" + "TRACKING_URI",
            "MLFLOW_" + "TRACKING_URI",
            "MLFLOW_" + "DLFS_URL",
            "MLFLOW_" + "COMPOSE_FILE",
            "infra/" + "mlflow",
            "artifacts/" + "runs",
        )
    )
    excluded = {
        ".git",
        ".venv",
        "references",
        "__pycache__",
        ".cache",
        ".staging",
        "data",
        "artifacts",
        ".pytest_cache",
        ".ruff_cache",
    }
    offenders: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excluded]
        for filename in filenames:
            if filename == ".env":
                continue
            path = Path(dirpath) / filename
            if path.stat().st_size > 500_000:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in retired:
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(root)}: {pattern.pattern}")
    assert not offenders, "retired tracking contracts remain:\n" + "\n".join(offenders)
