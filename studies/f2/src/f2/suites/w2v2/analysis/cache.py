"""W2V2 Evaluation cache and shared statistical helpers."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import yaml

from f2.suites.common.w2v.evaluate import verify_phrase_questions
from f2.suites.common.w2v.evaluation import parse_analogy_questions
from f2.suites.w2v1.resources import QUESTIONS_WORDS_SHA256

SEEDS = (1, 7, 19)
_EVALUATION_CACHE_SCHEMA = 2

# Explicit reconstruction protocol; keep the existing input-normalize-then-sum rule.
EVALUATION_PROTOCOL = {
    "analogy": "ASCII-uppercase; normalize inputs; b-a+c; valid-question denominator",
    "candidate_scope": "full checkpoint vocabulary; no frequency truncation",
    "query_exclusion": "exclude input token rows",
    "analogy_ties": "first vocabulary row; positive score required (compute-accuracy)",
    "neighbor_ties": "cosine descending, token bytes ascending, vocabulary row ascending",
    "zero_norm": "exclude zero-norm queries and candidates from nearest-neighbor results",
    "table5": "L2-normalize each input, sum, L2-normalize sum; reconstruction decision",
    "table5_model": "canonical Section 4.1 33B r3; no test-score model selection",
}


class W2V2EvaluationCache:
    """Signature-checked per-run cache for reconstructible W2V2 analogy & neighbor results."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(
        self,
        run_id: str,
        *,
        table: str,
        condition: str,
        questions_sha256: str | None,
        lookup_digest: str | None = None,
    ) -> dict[str, Any] | None:
        path = self._path(run_id, table)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            sig = payload.get("signature", {})
            if (
                sig.get("schema_version") != _EVALUATION_CACHE_SCHEMA
                or sig.get("table") != table
                or sig.get("condition") != condition
                or sig.get("questions_sha256") != questions_sha256
            ):
                return None
            if (
                lookup_digest is not None
                and sig.get("lookup_digest") is not None
                and sig.get("lookup_digest") != lookup_digest
            ):
                return None
            return payload.get("result")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return None

    def store(
        self,
        run_id: str,
        *,
        table: str,
        condition: str,
        questions_sha256: str | None,
        lookup_digest: str | None,
        result: dict[str, Any],
    ) -> None:
        try:
            path = self._path(run_id, table)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "signature": {
                            "schema_version": _EVALUATION_CACHE_SCHEMA,
                            "table": table,
                            "condition": condition,
                            "questions_sha256": questions_sha256,
                            "lookup_digest": lookup_digest,
                            "evaluation_protocol": EVALUATION_PROTOCOL,
                        },
                        "result": result,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    def _path(self, run_id: str, table: str) -> Path:
        safe_id = re.sub(r"[^0-9a-zA-Z_-]", "_", str(run_id))
        safe_tbl = re.sub(r"[^0-9a-zA-Z_-]", "_", str(table))
        return self.root / f"eval_{safe_tbl}_{safe_id}.json"


def summarize(values: list[float]) -> dict[str, Any]:
    """Two-sided 95% Student t interval for at most three independent seeds."""
    n = len(values)
    mean = statistics.mean(values) if n else None
    sd = statistics.stdev(values) if n > 1 else None
    margin = (
        ({2: 12.706204736, 3: 4.30265273}[n] * sd / math.sqrt(n)) if n > 1 else None
    )
    return {
        "n": n,
        "mean": mean,
        "standard_deviation": sd,
        "ci95_low": mean - margin if margin is not None else None,
        "ci95_high": mean + margin if margin is not None else None,
        "ci_method": "Student-t, two-sided, df=n-1",
    }


@functools.lru_cache(maxsize=1)
def _cached_targets():
    items = []
    config_dir = Path(__file__).parents[1] / "config"
    for path in sorted(config_dir.glob("e*.yaml")):
        raw = yaml.safe_load(path.read_text())
        raw["_path"] = path
        table = (
            "1"
            if path.name.startswith("e01")
            else "3"
            if path.name.startswith("e02")
            else "section4.1"
        )
        for variant in sorted(raw["variants"]):
            corpus, condition = variant.split("--", 1)
            items.append((table, corpus, condition, variant, raw))
    return tuple(items)


def targets():
    """Use the current canonical variant inventory, including reduced corpus budgets."""
    yield from _cached_targets()


def load_dataset_questions(path: Path | None, phrase: bool):
    if path is None or not Path(path).is_file():
        return {"status": "no_data", "reason": "evaluation data missing"}, ()
    p = Path(path)
    try:
        data = p.read_bytes()
        questions = parse_analogy_questions(data.splitlines())
        if not questions:
            return {
                "status": "no_data",
                "reason": "no valid analogy questions parsed",
            }, ()
        digest = hashlib.sha256(data).hexdigest()
        identity = {
            "questions_sha256": digest,
            "total_questions": len(questions),
        }
        if phrase:
            try:
                identity.update(verify_phrase_questions(p))
            except Exception:
                pass
        else:
            if digest == QUESTIONS_WORDS_SHA256 and len(questions) == 19544:
                identity["canonical"] = True
        return {"status": "complete", **identity}, questions
    except ValueError as exc:
        return {"status": "failed", "reason": str(exc)}, ()


def status_from_items(items):
    values = list(items)
    if values and all(s == "complete" for s in values):
        return "complete"
    for status in ("failed", "unverified", "killed", "running"):
        if status in values:
            return status
    return "incomplete" if "complete" in values or "incomplete" in values else "no_data"


__all__ = [
    "EVALUATION_PROTOCOL",
    "SEEDS",
    "W2V2EvaluationCache",
    "load_dataset_questions",
    "status_from_items",
    "summarize",
    "targets",
]
