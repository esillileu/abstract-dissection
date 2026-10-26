"""Read-only paper reporting over verified existing W2V2 result artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from itertools import pairwise
from pathlib import Path

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


def targets():
    """Use the current canonical variant inventory, including reduced corpus budgets."""
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
            yield table, corpus, condition, variant, raw


def _dataset(path, phrase):
    if path is None or not Path(path).is_file():
        return {"status": "no_data", "reason": "evaluation data missing"}, ()
    path = Path(path)
    try:
        data = path.read_bytes()
        questions = parse_analogy_questions(data.splitlines())
        if phrase:
            identity = verify_phrase_questions(path)
        else:
            if (
                hashlib.sha256(data).hexdigest() != QUESTIONS_WORDS_SHA256
                or len(questions) != 19544
            ):
                raise ValueError("word analogy data checksum/count mismatch")
            identity = {
                "questions_sha256": QUESTIONS_WORDS_SHA256,
                "total_questions": len(questions),
            }
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


def _load_run(record, config, variant, seed):
    """Read the published lookup; execution lineage is owned by the executor."""
    from f2.suites.w2v.artifacts import load_lookup_artifact

    if record.get("error") or not record.get("artifact_root"):
        raise EvidenceMissing(record.get("error", "cached artifacts unavailable"))
    root = Path(record["artifact_root"])
    result = json.loads((root / "result.json").read_text())
    lookup = root / _relative(result["artifacts"]["lookup"])
    if not lookup.resolve().is_relative_to(root.resolve()):
        raise ValueError("lookup escapes run cache")
    values = load_lookup_artifact(lookup)
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
        "training_corpus_sha256": values.manifest.get("corpus_digest"),
        "vocabulary_digest": values.manifest.get("vocabulary_digest"),
        "processed_tokens_by_epoch": [e["processed_tokens"] for e in result["epochs"]],
        "study_config_digest": identity.get("config_digest"),
        "engine_config_digest": values.manifest.get("config_digest"),
        "execution_plan_id": identity.get("execution_plan_id"),
        "experiment_spec_id": record["tags"].get("experiment_spec.id"),
        "atomic_variant": variant,
    }
    return values, result, root, inventory, provenance


def _timing(root, inventory, result):
    """Epoch-boundary resume histories are cumulative within each epoch, never summed twice."""
    try:
        hardware = json.loads(
            _verified_file(root, inventory, "environment/system.json").read_text()
        )
        if not all(hardware.get(k) for k in ("machine", "processor", "platform")):
            raise EvidenceMissing("hardware description incomplete")
        path = _verified_file(root, inventory, result["artifacts"]["metrics"])
        endpoints = {e["epoch"]: e["processed_tokens"] for e in result["epochs"]}
        observations = {}
        for row in csv.DictReader(path.open()):
            epoch, processed, elapsed = (
                int(row["epoch"]),
                int(row["processed_tokens"]),
                float(row["elapsed_seconds"]),
            )
            if epoch not in endpoints or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid time observation")
            key = epoch, processed
            if key in observations and observations[key] != elapsed:
                raise EvidenceMissing("conflicting resumed time observations")
            observations[key] = elapsed
        durations, linked = [], []
        previous_endpoint = 0
        for epoch, endpoint in endpoints.items():
            points = sorted((p, t) for (e, p), t in observations.items() if e == epoch)
            if (
                not points
                or points[-1][0] != endpoint
                or points[0][0] <= previous_endpoint
                or any(b[1] < a[1] for a, b in pairwise(points))
            ):
                raise EvidenceMissing("epoch time coverage/reset is unverifiable")
            durations.append(points[-1][1])
            linked.append(
                {
                    "epoch": epoch,
                    "processed_tokens": endpoint,
                    "elapsed_seconds": points[-1][1],
                    "hardware": hardware,
                }
            )
            previous_endpoint = endpoint
        return {
            "status": "complete",
            "minutes": sum(durations) / 60,
            "hardware": hardware,
            "verified_observation_count": len(observations),
            "observations": linked,
            "protocol": "sum per-epoch cumulative endpoint seconds; identical resumed rows deduplicated",
        }
    except (EvidenceMissing, KeyError, ValueError, OSError) as exc:
        return {"status": "unverified", "minutes": None, "reason": str(exc)}


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


def generate_report(
    *,
    records=(),
    paths=None,
    word_questions=None,
    phrase_questions=None,
    result_source="supplied records",
):
    """Evaluate only independently verified stored runs; report writing is not run completion."""
    paths = paths or RuntimePaths.from_environment()
    datasets = {
        "word": _dataset(word_questions, False),
        "phrase": _dataset(phrase_questions, True),
    }
    inventory = list(targets())
    selected, diagnostics = {}, []
    duplicates = set()
    run_owners = {}
    known = {(table, variant) for table, _, _, variant, _ in inventory}
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
                if key in selected:
                    duplicates.add(key)
                selected[key] = record
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
                "timing": {"status": "no_data", "minutes": None},
            }
            record = selected.get((table, variant, seed))
            lookup = None
            if record is not None:
                row.update(
                    run_id=record["run_id"],
                    mlflow_status=record.get("status", ""),
                    identity_status="unverified",
                )
                try:
                    if (table, variant, seed) in duplicates or (
                        table,
                        variant,
                        seed,
                    ) in conflicts:
                        raise ValueError(
                            "duplicate run/artifact identity across slots or seeds"
                        )
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
                        lookup, result, root, file_inventory, provenance = _load_run(
                            record, config, variant, seed
                        )
                        checkpoint = root / _relative(result["artifacts"]["lookup"])
                        row.update(
                            status="complete",
                            identity_status="complete",
                            reason="",
                            checkpoint=str(checkpoint),
                            checkpoint_reused=True,
                            provenance=provenance,
                        )
                        row["timing"] = _timing(root, file_inventory, result)
                        row["training_minutes"] = row["timing"]["minutes"]

                        if questions:
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
            for number, query, tokens, limit in _queries(table, condition):
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
                    try:
                        encoded = [t.encode() for t in tokens]
                        neighbors = (
                            additive_composition(lookup, encoded, limit=limit)
                            if number == "5"
                            else nearest_tokens(lookup, encoded[0], limit=limit)
                        )
                        output.update(
                            checkpoint_reused=True,
                            status="complete" if len(neighbors) == limit else "no_data",
                            reason=""
                            if len(neighbors) == limit
                            else "OOV, zero norm or insufficient valid candidates",
                            neighbors=[
                                {
                                    "token": n.token.decode("utf-8", errors="replace"),
                                    "cosine": n.score,
                                }
                                for n in neighbors
                            ],
                        )
                    except ValueError as exc:
                        output.update(status="failed", reason=str(exc))
                qualitative.append(output)
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
            ("total", "semantic", "syntactic", "training_minutes")
            if table == "1"
            else ("total",)
        ):

            def available(row, metric=metric):
                if row["identity_status"] != "complete":
                    return False
                return (
                    row["timing"]["status"] == "complete"
                    if metric == "training_minutes"
                    else row["metrics"].get(metric, {}).get("status") == "complete"
                )

            valid = [r for r in group if available(r)]
            reference = (
                dict(
                    zip(
                        ("training_minutes", "syntactic", "semantic", "total"),
                        paper.TABLE1[condition],
                        strict=True,
                    )
                )[metric]
                if table == "1"
                else group[0]["paper_reference"]
            )
            summary = summarize([r[metric] for r in valid])
            hardware = (
                [r["timing"]["hardware"] for r in valid]
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
                    "difference_from_paper": summary["mean"] - reference
                    if summary["mean"] is not None
                    else None,
                }
            )
    table_status = {}
    for number in ("1", "2", "3", "4", "5", "6", "section4.1"):
        if number == "2":
            states = [datasets["phrase"][0]["status"]]
        elif number in {"4", "5", "6"}:
            states = [q["status"] for q in qualitative if q["table"] == number]
        else:
            states = [a["status"] for a in aggregates if a["table"] == number]
        table_status[number] = _status(states)
    complete = not diagnostics and all(
        status == "complete" for status in table_status.values()
    )
    payload = {
        "result_source": result_source,
        "source": paper.SOURCE,
        "status": "complete" if complete else "incomplete",
        "report_generated": True,
        "experiment_complete": complete,
        "table_status": table_status,
        "quantitative_status": _status(a["status"] for a in aggregates),
        "qualitative_status": _status(q["status"] for q in qualitative),
        "identity_diagnostics_status": "failed" if diagnostics else "complete",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "seeds": SEEDS,
        "datasets": {k: v[0] for k, v in datasets.items()},
        "runs": rows,
        "aggregates": aggregates,
        "qualitative": qualitative,
        "diagnostics": diagnostics,
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
    output.mkdir(parents=True, exist_ok=True)
    table_exports = []
    for number in ("1", "2", "3", "4", "5", "6", "section4.1"):
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
    lines = [
        "# W2V2 paper reproduction",
        "",
        f"Source: {paper.SOURCE}",
        "",
        f"Result inventory: {result_source}. Missing entries describe supplied evidence, not proof that training never occurred.",
        "",
        f"Status: {payload['status']}. All current corpus variants are substitutes, not original Google news.",
        "",
        "95% Student-t intervals measure seed variation; n<2 has no interval. Completion does not imply matching paper accuracy.",
        "",
        f"Report generated: {payload['report_generated']}; experiment complete: {payload['experiment_complete']}",
        "",
        f"Table status: {json.dumps(table_status, sort_keys=True)}",
        "",
        "Evaluation protocol: " + json.dumps(EVALUATION_PROTOCOL, sort_keys=True),
    ]
    for table in ("1", "2", "3", "4", "5", "6", "section4.1"):
        lines += [
            "",
            f"## Table {table}" if table != "section4.1" else "## Section 4.1",
            "",
        ]
        data = (
            [a for a in aggregates if a["table"] == table]
            if table in {"1", "3", "section4.1"}
            else [q for q in qualitative if q["table"] == table]
        )
        if table == "2":
            lines += [
                f"Evaluation validation: {json.dumps(payload['datasets']['phrase'], sort_keys=True)}",
                "",
                "| Category | A | B | C | D |",
                "|---|---|---|---|---|",
            ]
            lines += ["| " + " | ".join(r) + " |" for r in paper.TABLE2]
        elif table in {"1", "3", "section4.1"}:
            lines += [
                "| Corpus | Condition | Metric | Paper | Mean | SD | n | 95% CI | Status | Missing seeds |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            lines += [
                f"| {a['corpus']} | {a['condition']} | {a['metric']} | {a['paper_reference']} | {a['mean']} | {a['standard_deviation']} | {a['n']} | {a['ci95_low']}, {a['ci95_high']} | {a['status']} | {a['missing_seeds']} |"
                for a in data
            ]
        else:
            if table == "6":
                lines += ["Reference columns: " + ", ".join(paper.TABLE6_QUERIES), ""]
            lines += [
                "Paper reference (never a reproduced observation):",
                "",
                "```json",
                json.dumps(payload["paper"][f"table{table}"], indent=2),
                "```",
                "",
                "| Corpus | Model | Seed | Run/checkpoint | Query | Neighbors | Status |",
                "|---|---|---|---|---|---|---|",
            ]
            lines += [
                f"| {q['corpus']} | {q['condition']} | {q['seed']} | {q['run_id']} / {q['checkpoint']} | {q['query']} | {json.dumps(q['neighbors'])} | {q['status']} |"
                for q in data
            ]
            if not data:
                lines += ["| — | — | — | — | — | — | no_data: missing checkpoint |"]
    lines += [
        "",
        "## Per-run evaluation denominators",
        "",
        "| Run | Metric | Total | Valid | Correct | Excluded | Coverage | Accuracy % | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['run_id']} | {name} | {m['total_count']} | {m['valid_count']} | {m['correct_count']} | {m['excluded_count']} | {m['coverage']} | {m['accuracy_percent']} | {m['status']} |"
        for r in rows
        for name, m in r["metrics"].items()
    ]
    lines += [
        "",
        "## Time comparability",
        "",
        "| Corpus | Condition | Environments | Comparable across seeds | Paper hardware |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {a['corpus']} | {a['condition']} | {json.dumps(a['hardware_environments'], sort_keys=True)} | {a['hardware_comparable_across_seeds']} | {a['paper_hardware_comparability']} |"
        for a in aggregates
        if a["metric"] == "training_minutes" and a["n"]
    ]
    lines += [
        "",
        "## Run evidence and timing",
        "",
        "| Run | Identity | Provenance | Timing |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r.get('run_id', '')} | {r['identity_status']} | {json.dumps(r.get('provenance', {}), sort_keys=True)} | {json.dumps(r['timing'], sort_keys=True)} |"
        for r in rows
        if r.get("run_id")
    ]
    lines += [
        "",
        "## Missing or failed conditions",
        "",
        "| Table | Corpus | Condition | Seed | Status | Reason |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['table']} | {r['corpus']} | {r['condition']} | {r['seed']} | {r['status']} | {r['reason']} |"
        for r in rows
        if r["status"] != "complete"
    ]
    lines += [
        "",
        "External Table 6 baselines: reference-only; retraining unsupported.",
        "",
        "Diagnostics: " + json.dumps(diagnostics, sort_keys=True),
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return output
