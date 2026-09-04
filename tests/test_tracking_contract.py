from __future__ import annotations

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
    offenders: list[str] = []
    for path in root.rglob("*"):
        if (
            path.name == ".env"
            or not path.is_file()
            or any(
                part in {".git", ".venv", "references", "__pycache__"}
                for part in path.parts
            )
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in retired:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(root)}: {pattern.pattern}")
    assert not offenders, "retired tracking contracts remain:\n" + "\n".join(offenders)
