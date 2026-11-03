"""Reusable evaluation cache base for W2V1 tables."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


class BaseEvaluationCache:
    """Signature-checked per-run cache for reconstructible analogy and evaluation results."""

    def __init__(self, root: Path, signature: dict[str, Any], result_cls: type) -> None:
        self.root = root
        self.signature = signature
        self.result_cls = result_cls

    def _path(self, run_id: str) -> Path:
        return self.root / f"eval_{run_id}.json"

    def load(self, run_id: str) -> Any:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if payload.get("signature") != self.signature:
                return None
            result_data = payload.get("result")
            if not isinstance(result_data, dict):
                return None
            return self.result_cls(**result_data)
        except Exception:
            return None

    def store(self, run_id: str, result: Any) -> None:
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": self.signature,
            "result": asdict(result),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = ["BaseEvaluationCache"]
