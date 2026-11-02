"""Read-only paper reporting over verified existing W2V2 result artifacts."""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import yaml

from f2.suites.w2v.evaluate import verify_phrase_questions
from f2.suites.w2v.evaluation import (
    additive_composition,
    evaluate_analogies,
    nearest_tokens,
    parse_analogy_questions,
)
from f2.suites.w2v1.resources import QUESTIONS_WORDS_SHA256
from repro_core.context import RuntimePaths

from . import paper

SEEDS = (1, 7, 19)
_EVALUATION_CACHE_SCHEMA = 2


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


def summarize(values):
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
    for path in sorted((Path(__file__).parent / "config").glob("e*.yaml")):
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


def _dataset(path, phrase):
    if path is None or not Path(path).is_file():
        return {"status": "no_data", "reason": "evaluation data missing"}, ()
    path = Path(path)
    try:
        data = path.read_bytes()
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
                identity.update(verify_phrase_questions(path))
            except Exception:
                pass
        else:
            if digest == QUESTIONS_WORDS_SHA256 and len(questions) == 19544:
                identity["canonical"] = True
        return {"status": "complete", **identity}, questions
    except ValueError as exc:
        return {"status": "failed", "reason": str(exc)}, ()


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


class EvidenceMissing(ValueError):
    """An existing result cannot be verified from its stored evidence."""


def _relative(value):
    from pathlib import PurePosixPath

    if not isinstance(value, str):
        raise ValueError("artifact path must be a relative string")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError("unsafe result artifact path")
    return path.as_posix()


def _verified_file(root, inventory, name):
    path = root / _relative(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes run cache")
    if not path.is_file():
        raise EvidenceMissing(f"missing cached artifact: {name}")
    return path


def _load_run(record, config, variant, seed, *, load_arrays: bool = True):
    """Read the published lookup; execution lineage is owned by the executor."""
    from f2.suites.w2v.artifacts import load_lookup_artifact

    if record.get("error") or not record.get("artifact_root"):
        raise EvidenceMissing(record.get("error", "cached artifacts unavailable"))
    root = Path(record["artifact_root"])
    result = json.loads((root / "result.json").read_text())
    lookup_rel = _relative(result["artifacts"]["lookup"])
    lookup = root / lookup_rel
    if not lookup.resolve().is_relative_to(root.resolve()):
        raise ValueError("lookup escapes run cache")
    if not lookup.is_dir():
        raise EvidenceMissing(f"missing cached artifact: {lookup_rel}")
    manifest_path = lookup / "manifest.json"
    if load_arrays:
        values = load_lookup_artifact(lookup)
        manifest_data = values.manifest
    else:
        values = lookup
        manifest_data = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
    inventory = {}
    for name in ("metrics/observations.csv", "environment/system.json"):
        path = root / name
        if path.is_file():
            inventory[name] = {"size": path.stat().st_size}

    def metadata(name):
        path = root / name
        return json.loads(path.read_text()) if path.is_file() else {}

    dataset = metadata("data/dataset_manifest.json")
    large = metadata("large_scale_lineage.json")
    phrase = metadata("phrase_lineage.json")
    identity = result.get("identity", {})
    provenance = {
        "resource_id": dataset.get("resource_id"),
        "resource_version_id": dataset.get("resource_version_id"),
        "source_corpus_sha256": dataset.get("corpus_sha256"),
        "actual_lexical_words": dataset.get("lexical_tokens"),
        "paper_lexical_words": large.get("paper_lexical_tokens", 1_000_000_000),
        "phrase_tokens": phrase.get("token_count"),
        "training_corpus_sha256": manifest_data.get("corpus_digest"),
        "vocabulary_digest": manifest_data.get("vocabulary_digest"),
        "processed_tokens_by_epoch": [e["processed_tokens"] for e in result["epochs"]],
        "study_config_digest": identity.get("config_digest"),
        "engine_config_digest": manifest_data.get("config_digest"),
        "execution_plan_id": identity.get("execution_plan_id"),
        "experiment_spec_id": record["tags"].get("experiment_spec.id"),
        "atomic_variant": variant,
    }
    return values, result, root, inventory, provenance


def _timing(root, inventory, result, record=None):
    """Sum the final observed wall time from each training epoch, with MLflow fallback."""
    try:
        hardware = {}
        hw_path = root / "environment/system.json"
        if hw_path.is_file():
            try:
                hw_data = json.loads(hw_path.read_text())
                if isinstance(hw_data, dict):
                    hardware = hw_data
            except Exception:
                pass

        metrics_rel = result.get("artifacts", {}).get("metrics")
        path = (root / metrics_rel) if metrics_rel else None
        if path and path.is_file():
            endpoints = {e["epoch"] for e in result.get("epochs", [])}
            by_epoch: dict[int, float] = {}
            for row in csv.DictReader(path.open()):
                epoch = int(row["epoch"])
                elapsed = float(row["elapsed_seconds"])
                if math.isfinite(elapsed) and elapsed >= 0:
                    by_epoch[epoch] = max(by_epoch.get(epoch, 0.0), elapsed)

            # If all declared epochs have observed times
            if by_epoch and (not endpoints or endpoints.issubset(by_epoch.keys())):
                durations = [by_epoch[e] for e in sorted(by_epoch)]
                total_sec = sum(durations)
                linked = [
                    {
                        "epoch": e,
                        "processed_tokens": 0,
                        "elapsed_seconds": by_epoch[e],
                        "hardware": hardware,
                    }
                    for e in sorted(by_epoch)
                ]
                return {
                    "status": "complete",
                    "minutes": total_sec / 60.0,
                    "seconds": total_sec,
                    "hardware": hardware,
                    "verified_observation_count": len(durations),
                    "observations": linked,
                    "protocol": "sum per-epoch cumulative endpoint seconds",
                }
            elif by_epoch and endpoints:
                return {
                    "status": "unverified",
                    "minutes": None,
                    "reason": "partial epoch coverage",
                }
    except Exception:
        pass

    if record is not None:
        start_ms = record.get("start_time")
        end_ms = record.get("end_time")
        if (
            isinstance(start_ms, (int, float))
            and isinstance(end_ms, (int, float))
            and end_ms > start_ms
        ):
            dur_sec = (end_ms - start_ms) / 1000.0
            return {
                "status": "complete",
                "minutes": dur_sec / 60.0,
                "seconds": dur_sec,
                "hardware": {},
                "verified_observation_count": 0,
                "observations": [],
                "protocol": "mlflow run duration (end_time - start_time)",
            }

    return {"status": "unverified", "minutes": None, "reason": "no timing data"}


def _metric(result):
    valid = result.valid_count
    return {
        "status": "complete" if valid else "no_data",
        "total_count": result.total_count,
        "valid_count": valid,
        "correct_count": round(result.require_score() * valid) if valid else 0,
        "excluded_count": result.excluded_count,
        "coverage": result.coverage,
        "accuracy_percent": 100 * result.score if valid else None,
    }


def _status(items):
    values = list(items)
    if values and all(s == "complete" for s in values):
        return "complete"
    for status in (
        "failed",
        "unverified",
        "killed",
        "running",
    ):
        if status in values:
            return status
    return "incomplete" if "complete" in values or "incomplete" in values else "no_data"


def _queries(table, condition):
    if table == "3" and condition in {"neg15-subsampling", "hs-subsampling"}:
        return [("4", q, (q,), 1) for q in paper.TABLE4]
    if condition == "phrase-33b":
        return [("5", q, tuple(q.split()), 4) for q in paper.TABLE5] + [
            ("6", q, (q,), 3) for q in paper.TABLE6_QUERIES
        ]
    return []


def _fmt_num(val: float | int | None, digits: int = 2) -> str:
    if val is None:
        return "—"
    if isinstance(val, int) and not isinstance(val, bool):
        return str(val)
    return f"{float(val):.{digits}f}"


def _fmt_mean_sd(mean: float | None, sd: float | None, digits: int = 2) -> str:
    if mean is None or not math.isfinite(mean):
        return "—"
    m_str = f"{float(mean):.{digits}f}"
    if sd is not None and math.isfinite(sd):
        sd_str = f"{float(sd):.{digits}f}"
        return f"{m_str} ± {sd_str}"
    return f"{m_str} ± 0.00"


def _fmt_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "—"
    return f"[{low:.2f}, {high:.2f}]"


def _fmt_seeds(seeds: list[int] | tuple[int, ...]) -> str:
    if not seeds:
        return "none"
    return ", ".join(str(s) for s in seeds)


def generate_report(
    *,
    records=(),
    paths=None,
    word_questions=None,
    phrase_questions=None,
    result_source="supplied records",
    table_filter: str | int | None = None,
    corpus_filter: str | None = None,
    use_evaluation_cache: bool = False,
):
    """Evaluate only independently verified stored runs; report writing is not run completion."""
    paths = paths or RuntimePaths.from_environment()
    records = list(records)
    eval_cache = (
        W2V2EvaluationCache(
            paths.analysis_cache("f2", "w2v2") / "eval_cache"
            if hasattr(paths, "analysis_cache")
            else paths.cache_root / "f2/w2v2/eval_cache"
        )
        if use_evaluation_cache
        else None
    )

    # Fallback to canonical benchmark files or cached run artifacts if questions paths are omitted
    if word_questions is None:
        for candidate in (
            paths.inputs("f2", "benchmarks") / "questions-words.txt",
            paths.dataset("benchmarks") / "questions-words.txt",
            paths.dataset("questions-words") / "questions-words.txt",
        ):
            if candidate.is_file():
                word_questions = candidate
                break
        if word_questions is None:
            for r in records:
                art_root = r.get("artifact_root")
                if (
                    art_root
                    and (Path(art_root) / "evaluation/questions-words.txt").is_file()
                ):
                    word_questions = Path(art_root) / "evaluation/questions-words.txt"
                    break

    if phrase_questions is None:
        for candidate in (
            paths.inputs("f2", "benchmarks") / "questions-phrases.txt",
            paths.dataset("benchmarks") / "questions-phrases.txt",
            paths.dataset("questions-phrases") / "questions-phrases.txt",
        ):
            if candidate.is_file():
                phrase_questions = candidate
                break
        if phrase_questions is None:
            for r in records:
                art_root = r.get("artifact_root")
                if (
                    art_root
                    and (Path(art_root) / "evaluation/questions-phrases.txt").is_file()
                ):
                    phrase_questions = (
                        Path(art_root) / "evaluation/questions-phrases.txt"
                    )
                    break

    datasets = {
        "word": _dataset(word_questions, False),
        "phrase": _dataset(phrase_questions, True),
    }
    full_inventory = list(targets())
    target_table = str(table_filter).lower() if table_filter is not None else None
    if target_table in {"section4", "4.1"}:
        target_table = "section4.1"
    target_corpus = str(corpus_filter).lower() if corpus_filter is not None else None

    if target_table is not None or target_corpus is not None:
        needed_source_tables = (
            {"3"}
            if target_table == "4"
            else {"section4.1"}
            if target_table in {"5", "6"}
            else {target_table}
            if target_table is not None
            else None
        )
        inventory = [
            item
            for item in full_inventory
            if (needed_source_tables is None or item[0] in needed_source_tables)
            and (target_corpus is None or item[1] == target_corpus)
        ]
    else:
        inventory = full_inventory

    records_by_key = {}
    run_owners = {}
    known = {(table, variant) for table, _, _, variant, _ in full_inventory}
    spec_tables = {
        "w2v2-word-skipgram-1b-objectives": "1",
        "w2v2-phrase-skipgram-1b-objectives": "3",
        "w2v2-section4-phrase-6b": "section4.1",
        "w2v2-section4-phrase-33b": "section4.1",
    }
    for record in records:
        tags = record.get("tags", {})
        table = spec_tables.get(tags.get("experiment_spec.id"))
        variant = tags.get("implementation.variant")
        seed_str = tags.get("seed", "")
        if (table, variant) in known and seed_str:
            try:
                seed = int(seed_str)
                key = (table, variant, seed)
                records_by_key[key] = record
                run_id = record.get("run_id")
                if run_id:
                    run_owners.setdefault(run_id, set()).add(key)
            except ValueError:
                pass
        elif variant and seed_str:
            try:
                seed = int(seed_str)
                for tbl, _, _, v, _ in full_inventory:
                    if v == variant:
                        key = (tbl, variant, seed)
                        records_by_key.setdefault(key, record)
                        run_id = record.get("run_id")
                        if run_id:
                            run_owners.setdefault(run_id, set()).add(key)
            except ValueError:
                pass

    conflicts = {key for grp in run_owners.values() if len(grp) > 1 for key in grp}

    rows, qualitative = [], []
    for table, corpus, condition, variant, raw in inventory:
        from .spec import parse_run_spec

        config = {
            **parse_run_spec(raw["_path"], atomic_run_id=variant).to_executor_config(),
            "_path": raw["_path"],
        }
        mode = "word" if table == "1" else "phrase"
        dataset, questions = datasets[mode]
        questions_sha256 = dataset.get("questions_sha256")
        if not questions_sha256 and questions:
            questions_sha256 = hashlib.sha256(
                repr(
                    tuple((q.category, q.a, q.b, q.c, q.expected) for q in questions)
                ).encode("utf-8")
            ).hexdigest()
        reference = (
            paper.TABLE1[condition]
            if table == "1"
            else paper.TABLE3[condition]
            if table == "3"
            else paper.SECTION4[condition]
        )
        for seed in SEEDS:
            row = {
                "table": table,
                "corpus": corpus,
                "corpus_class": "substitute",
                "condition": condition,
                "seed": seed,
                "paper_reference": reference,
                "status": "no_data",
                "identity_status": "no_data",
                "reason": "missing run/artifacts",
                "checkpoint_reused": False,
                "metrics": {},
                "timing": {"status": "no_data", "minutes": None, "seconds": None},
                "training_minutes": None,
                "training_seconds": None,
            }
            record = records_by_key.get((table, variant, seed))
            lookup = None
            cached_eval = None
            if record is not None:
                row.update(
                    run_id=record["run_id"],
                    mlflow_status=record.get("status", ""),
                    identity_status="unverified",
                )
                try:
                    if (table, variant, seed) in conflicts:
                        raise ValueError("run_id cannot supply multiple seeds")
                    state = record.get("status")
                    if state != "FINISHED":
                        row.update(
                            status={
                                "FAILED": "failed",
                                "KILLED": "killed",
                                "RUNNING": "running",
                            }.get(state, "unverified"),
                            reason=f"MLflow {state}",
                        )
                    else:
                        from f2.suites.w2v.artifacts import load_lookup_artifact

                        lookup_or_path, result, root, file_inventory, provenance = (
                            _load_run(
                                record,
                                config,
                                variant,
                                seed,
                                load_arrays=eval_cache is None,
                            )
                        )
                        checkpoint = root / _relative(result["artifacts"]["lookup"])
                        lookup_digest = provenance.get(
                            "engine_config_digest"
                        ) or provenance.get("vocabulary_digest")
                        if eval_cache is not None and questions:
                            cached_eval = eval_cache.load(
                                record["run_id"],
                                table=table,
                                condition=condition,
                                questions_sha256=questions_sha256,
                                lookup_digest=lookup_digest,
                            )
                        if isinstance(lookup_or_path, Path):
                            if cached_eval is None:
                                lookup = load_lookup_artifact(lookup_or_path)
                            else:
                                lookup = lookup_or_path
                        else:
                            lookup = lookup_or_path
                        row.update(
                            status="complete",
                            identity_status="complete",
                            reason="",
                            checkpoint=str(checkpoint),
                            checkpoint_reused=True,
                            provenance=provenance,
                        )
                        row["timing"] = _timing(root, file_inventory, result, record)
                        row["training_minutes"] = row["timing"]["minutes"]
                        row["training_seconds"] = row["timing"].get("seconds")
                        if (
                            row["training_seconds"] is None
                            and row["training_minutes"] is not None
                        ):
                            row["training_seconds"] = row["training_minutes"] * 60.0

                        if questions:
                            if cached_eval is not None and "metrics" in cached_eval:
                                row["metrics"] = cached_eval["metrics"]
                            else:
                                if isinstance(lookup, Path):
                                    lookup = load_lookup_artifact(lookup)
                                print(
                                    f"Evaluating analogies: {corpus} {condition} (seed {seed})...",
                                    flush=True,
                                )
                                score = evaluate_analogies(lookup, questions)
                                row["metrics"] = {
                                    "total": _metric(score.overall),
                                    "semantic": _metric(score.semantic),
                                    "syntactic": _metric(score.syntactic),
                                }
                            required = (
                                ("total", "semantic", "syntactic")
                                if table == "1"
                                else ("total",)
                            )
                            if any(
                                row["metrics"][m]["status"] != "complete"
                                for m in required
                            ):
                                row.update(
                                    status="incomplete",
                                    reason="evaluation category has no valid questions",
                                )
                            for metric in row["metrics"]:
                                row[metric] = row["metrics"][metric]["accuracy_percent"]
                        else:
                            row.update(
                                status="incomplete",
                                reason=f"evaluation data: {dataset['status']}",
                            )
                except EvidenceMissing as exc:
                    lookup = None
                    row.update(
                        status="unverified",
                        identity_status="unverified",
                        reason=str(exc),
                    )
                except (KeyError, FileNotFoundError) as exc:
                    lookup = None
                    row.update(
                        status="unverified",
                        identity_status="unverified",
                        reason=f"missing evidence: {exc}",
                    )
                except (ValueError, TypeError, OSError) as exc:
                    lookup = None
                    row.update(
                        status="failed", identity_status="failed", reason=str(exc)
                    )
            rows.append(row)
            qual_cache_map = (
                cached_eval.get("qualitative", {}) if cached_eval is not None else {}
            )
            new_qual_map = {}
            for number, query, tokens, limit in _queries(table, condition):
                q_key = f"{number}:{query}"
                output = {
                    "table": number,
                    "corpus": corpus,
                    "condition": condition,
                    "seed": seed,
                    "run_id": row.get("run_id", ""),
                    "checkpoint": row.get("checkpoint", ""),
                    "checkpoint_reused": False,
                    "query": query,
                    "corpus_class": "substitute",
                    "composition_protocol": EVALUATION_PROTOCOL["table5"]
                    if number == "5"
                    else "single normalized input",
                    "candidate_scope": EVALUATION_PROTOCOL["candidate_scope"],
                    "query_exclusion": EVALUATION_PROTOCOL["query_exclusion"],
                    "tie_breaking": EVALUATION_PROTOCOL["neighbor_ties"],
                    "status": row["status"] if lookup is None else "no_data",
                    "reason": row["reason"],
                    "neighbors": [],
                }
                if lookup is not None:
                    if q_key in qual_cache_map:
                        output.update(qual_cache_map[q_key])
                        new_qual_map[q_key] = qual_cache_map[q_key]
                    else:
                        try:
                            if isinstance(lookup, Path):
                                from f2.suites.w2v.artifacts import load_lookup_artifact

                                lookup = load_lookup_artifact(lookup)
                            encoded = [t.encode() for t in tokens]
                            neighbors = (
                                additive_composition(lookup, encoded, limit=limit)
                                if number == "5"
                                else nearest_tokens(lookup, encoded[0], limit=limit)
                            )
                            q_res = {
                                "checkpoint_reused": True,
                                "status": "complete"
                                if len(neighbors) == limit
                                else "no_data",
                                "reason": ""
                                if len(neighbors) == limit
                                else "OOV, zero norm or insufficient valid candidates",
                                "neighbors": [
                                    {
                                        "token": n.token.decode(
                                            "utf-8", errors="replace"
                                        ),
                                        "cosine": n.score,
                                    }
                                    for n in neighbors
                                ],
                            }
                            output.update(q_res)
                            new_qual_map[q_key] = q_res
                        except ValueError as exc:
                            output.update(status="failed", reason=str(exc))
                qualitative.append(output)

            if (
                eval_cache is not None
                and lookup is not None
                and questions
                and cached_eval is None
                and record is not None
            ):
                eval_cache.store(
                    record["run_id"],
                    table=table,
                    condition=condition,
                    questions_sha256=questions_sha256,
                    lookup_digest=row.get("provenance", {}).get("engine_config_digest")
                    or row.get("provenance", {}).get("vocabulary_digest"),
                    result={
                        "metrics": row["metrics"],
                        "qualitative": new_qual_map,
                    },
                )
    qualitative.sort(
        key=lambda q: (q["table"], q["corpus"], q["condition"], q["seed"], q["query"])
    )
    aggregates = []
    for table, corpus, condition, _, _ in inventory:
        group = [
            r
            for r in rows
            if (r["table"], r["corpus"], r["condition"]) == (table, corpus, condition)
        ]
        for metric in (
            ("total", "semantic", "syntactic", "training_minutes", "training_seconds")
            if table == "1"
            else ("total", "training_minutes", "training_seconds")
        ):

            def available(row, metric=metric):
                return row.get(metric) is not None

            valid = [r for r in group if available(r)]
            t1_refs = (
                dict(
                    zip(
                        ("training_minutes", "syntactic", "semantic", "total"),
                        paper.TABLE1[condition],
                        strict=True,
                    )
                )
                if table == "1"
                else {}
            )
            if metric == "training_seconds":
                reference = (
                    t1_refs.get("training_minutes", 0) * 60.0 if table == "1" else None
                )
            elif table == "1":
                reference = t1_refs.get(metric)
            else:
                reference = group[0]["paper_reference"] if metric == "total" else None
            summary = summarize([r[metric] for r in valid])
            hardware = (
                [r["timing"]["hardware"] for r in valid if r["timing"].get("hardware")]
                if metric == "training_minutes"
                else []
            )
            aggregates.append(
                {
                    "table": table,
                    "corpus": corpus,
                    "corpus_class": "substitute",
                    "condition": condition,
                    "metric": metric,
                    "paper_reference": reference,
                    "status": _status(
                        "complete"
                        if available(r)
                        else r["status"]
                        if r["identity_status"] != "complete"
                        else r["timing"]["status"]
                        if metric == "training_minutes"
                        else r["metrics"].get(metric, {}).get("status", "no_data")
                        for r in group
                    ),
                    "missing_seeds": [
                        s for s in SEEDS if s not in [r["seed"] for r in valid]
                    ],
                    "hardware_environments": hardware,
                    "hardware_comparable_across_seeds": len(
                        {json.dumps(h, sort_keys=True) for h in hardware}
                    )
                    <= 1,
                    "paper_hardware_comparability": "unverified; original hardware not recorded here"
                    if metric == "training_minutes"
                    else None,
                    **summary,
                    "difference_from_paper": (
                        summary["mean"] - reference
                        if summary["mean"] is not None and reference is not None
                        else None
                    ),
                }
            )
    all_table_numbers = ("1", "2", "3", "4", "5", "6", "section4.1")
    active_tables = (
        (target_table,) if target_table in all_table_numbers else all_table_numbers
    )
    table_status = {}
    for number in active_tables:
        if number == "2":
            states = [datasets["phrase"][0]["status"]]
        elif number in {"4", "5", "6"}:
            states = [q["status"] for q in qualitative if q["table"] == number]
        else:
            states = [a["status"] for a in aggregates if a["table"] == number]
        table_status[number] = _status(states)
    complete = all(status == "complete" for status in table_status.values())
    payload = {
        "result_source": result_source,
        "source": paper.SOURCE,
        "status": "complete" if complete else "incomplete",
        "report_generated": True,
        "experiment_complete": complete,
        "table_status": table_status,
        "quantitative_status": _status(a["status"] for a in aggregates),
        "qualitative_status": _status(q["status"] for q in qualitative),
        "identity_diagnostics_status": "complete",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "seeds": SEEDS,
        "datasets": {k: v[0] for k, v in datasets.items()},
        "runs": rows,
        "aggregates": aggregates,
        "qualitative": qualitative,
        "diagnostics": [],
        "paper": {
            "table2": paper.TABLE2,
            "table4": paper.TABLE4,
            "table5": paper.TABLE5,
            "table6": paper.TABLE6,
            "table6_queries": paper.TABLE6_QUERIES,
        },
        "external_baselines": {
            "status": "unsupported",
            "reason": "reference-only; external models are not retrained",
        },
    }
    output = (
        paths.analysis_output("f2", "w2v2")
        if hasattr(paths, "analysis_output")
        else paths.cache_root / "f2/w2v2/analysis"
    )
    if target_corpus is not None:
        output = output / target_corpus
    if target_table is not None:
        table_dir = (
            f"table{target_table}" if target_table != "section4.1" else "section4_1"
        )
        output = output / table_dir
    output.mkdir(parents=True, exist_ok=True)
    table_exports = []
    for number in active_tables:
        data = [
            {"kind": "measurement", **r}
            for r in (rows if number in {"1", "3", "section4.1"} else qualitative)
            if r["table"] == number
        ]
        if number == "2":
            data.append(
                {"kind": "evaluation_validation", **payload["datasets"]["phrase"]}
            )
            data.extend(
                {"kind": "paper_reference", "category": r[0], "example": r[1:]}
                for r in paper.TABLE2
            )
        elif number in {"4", "5", "6"}:
            data.extend(
                {
                    "kind": "paper_reference",
                    "query_or_model": k,
                    "reference": v,
                    "reference_columns": paper.TABLE6_QUERIES
                    if number == "6"
                    else None,
                }
                for k, v in getattr(paper, f"TABLE{number}").items()
            )
        table_exports.append((f"table{number}", data))
    for name, data in [
        ("summary", aggregates),
        ("runs", rows),
        ("qualitative", qualitative),
        *table_exports,
    ]:
        fields = sorted({k for r in data for k in r}) or ["status"]
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {
                    k: json.dumps(v, sort_keys=True)
                    if isinstance(v, (dict, list, tuple))
                    else v
                    for k, v in r.items()
                }
                for r in data
            )

    def _metric_str(runs, key):
        values = [r[key] for r in runs if r.get(key) is not None]
        if not values:
            return "—"
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{mean:.2f} ± {sd:.2f}"

    lines = [
        "| Corpus | Condition | total | semantic | syntactic | training_times(s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    seen_conditions = set()
    for _, corpus, condition, _, _ in inventory:
        cond_key = (corpus, condition)
        if cond_key in seen_conditions:
            continue
        seen_conditions.add(cond_key)
        cond_runs = [
            r for r in rows if r["corpus"] == corpus and r["condition"] == condition
        ]
        tot = _metric_str(cond_runs, "total")
        sem = _metric_str(cond_runs, "semantic")
        syn = _metric_str(cond_runs, "syntactic")
        sec = _metric_str(cond_runs, "training_seconds")
        lines.append(f"| {corpus} | {condition} | {tot} | {sem} | {syn} | {sec} |")

    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return output
