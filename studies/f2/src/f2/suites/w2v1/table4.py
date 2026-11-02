"""Full-vocabulary Table 4 analogy analysis over durable canonical runs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.cache import LocalResults, unavailable_report
from f2.suites.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths

from .analysis import CORPUS_SOURCES, observed_training_seconds

_CONDITION = re.compile(r"(wmt|lm1b|umbc)--(cbow|skipgram)-d300-w783m")
_NNLM_CONDITION = re.compile(r"fineweb--nnlm-d(20|50|100)-w6000m")
TABLE4_CORPUS_SOURCES = (*CORPUS_SOURCES, "fineweb")
_SEEDS = {1, 7, 19}
_REFERENCE = {
    "cbow": (15.5, 53.1, 36.1),
    "skipgram": (50.0, 55.9, 53.3),
    "nnlm-d20": (12.9, 26.4, 20.3),
    "nnlm-d50": (27.9, 55.8, 43.2),
    "nnlm-d100": (34.2, 64.5, 50.8),
}
_EXTERNAL_REFERENCE = (
    ("Collobert-Weston NNLM", 50, "660M", 9.3, 12.3, 11.0),
    ("Turian NNLM", 50, "37M", 1.4, 2.6, 2.1),
    ("Turian NNLM", 200, "37M", 1.4, 2.2, 1.8),
    ("Mnih NNLM", 50, "37M", 1.8, 9.1, 5.8),
    ("Mnih NNLM", 100, "37M", 3.3, 13.2, 8.8),
    ("Mikolov RNNLM", 80, "320M", 4.9, 18.4, 12.7),
    ("Mikolov RNNLM", 640, "320M", 8.6, 36.5, 24.6),
    ("Huang NNLM", 50, "990M", 13.3, 11.6, 12.3),
)


@dataclass(frozen=True)
class Table4RunResult:
    architecture: str
    corpus: str
    seed: int
    mlflow_run_id: str
    semantic_accuracy_percent: float
    syntactic_accuracy_percent: float
    total_accuracy_percent: float
    included_questions: int
    total_questions: int
    observed_training_seconds: float


_EVALUATION_PROTOCOL = "table4-v1"


class Table4EvaluationCache:
    """Signature-checked per-run cache for reconstructible Table 4 analogy results."""

    def __init__(self, root: Path, *, questions_sha256: str) -> None:
        self.root = root
        self.signature = {
            "schema_version": 2,
            "evaluation_protocol": _EVALUATION_PROTOCOL,
            "questions_sha256": questions_sha256,
        }

    def load(self, run_id: str) -> Table4RunResult | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != self.signature:
                return None
            result = Table4RunResult(**payload["result"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return result if result.mlflow_run_id == run_id else None

    def store(self, result: Table4RunResult) -> None:
        path = self._path(result.mlflow_run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "signature": self.signature,
                    "result": asdict(result),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _path(self, run_id: str) -> Path:
        if re.fullmatch(r"[0-9a-zA-Z_-]+", run_id) is None:
            raise ValueError(f"invalid MLflow run ID: {run_id}")
        return self.root / f"{run_id}.json"


def complete_conditions(runs: list[Any], corpus: str) -> dict[str, dict[int, Any]]:
    grouped: dict[str, dict[int, Any]] = {}
    for run in runs:
        tags = run.data.tags
        variant = tags.get("implementation.variant", "")
        nnlm = _NNLM_CONDITION.fullmatch(variant)
        if nnlm is not None and corpus == "fineweb":
            dimension = nnlm.group(1)
            architecture = f"nnlm-d{dimension}"
            if tags.get("experiment_spec.id") != "w2v1-google-news-nnlm-6b":
                continue
            seed = int(tags.get("seed", "0"))
            slot = f"w2v1-fineweb-reconstruction-r1-nnlm-d{dimension}-w6000m-s{seed}"
        else:
            match = _CONDITION.fullmatch(variant)
            if match is None or match.group(1) != corpus:
                continue
            architecture = match.group(2)
            if (
                tags.get("experiment_spec.id")
                != f"w2v1-google-news-{architecture}-scale"
            ):
                continue
            seed = int(tags.get("seed", "0"))
            slot = f"w2v1-table4-{corpus}-r1-{architecture}-d300-w783m-s{seed}"
        if seed not in _SEEDS or run.data.tags.get("f2.planned_run_slot_id") != slot:
            continue
        grouped.setdefault(architecture, {}).setdefault(seed, run)
    return {key: value for key, value in grouped.items() if set(value) == _SEEDS}


def analyze_table4_sources(
    cache_root: Path,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:

    paths = paths or RuntimePaths.from_environment()
    if not Path(questions_path).is_file():
        output = unavailable_report(
            paths.cache_root / "f2/w2v1/analysis/table4",
            "cached evaluation questions missing",
        )
        return (output,)
    if corpus_source is not None and corpus_source not in TABLE4_CORPUS_SOURCES:
        raise ValueError(f"unsupported W2V1 corpus source: {corpus_source}")
    questions_path = Path(questions_path)
    question_bytes = questions_path.read_bytes()
    questions = parse_analogy_questions(question_bytes.splitlines())
    if not questions:
        raise ValueError("questions-words.txt must contain at least one question")
    question_digest = hashlib.sha256(question_bytes).hexdigest()
    artifact_cache = LocalResults(cache_root)
    runs = artifact_cache.runs()

    evaluation_cache = Table4EvaluationCache(
        paths.cache_root / "f2/w2v1/table4", questions_sha256=question_digest
    )
    sources = TABLE4_CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    outputs = []
    for corpus in sources:
        output = (paths.cache_root / "f2/w2v1/analysis") / corpus / "table4"
        output.mkdir(parents=True, exist_ok=True)
        records = []
        unavailable = []
        cache_hits = 0
        cache_misses = 0
        for architecture, condition_runs in sorted(
            complete_conditions(runs, corpus).items()
        ):
            for seed, run in sorted(condition_runs.items()):
                cached = evaluation_cache.load(run.info.run_id)
                if (
                    cached is not None
                    and cached.seed == seed
                    and cached.architecture == architecture
                    and cached.corpus == corpus
                ):
                    records.append(cached)
                    cache_hits += 1
                    continue
                try:
                    cache_misses += 1
                    lookup = load_lookup_artifact(
                        artifact_cache.get(run.info.run_id, "lookup")
                    )
                    result = evaluate_analogies(
                        lookup, questions, vocabulary_limit=None, batch_size=256
                    )
                    seconds = observed_training_seconds(
                        artifact_cache.get(run.info.run_id, "metrics/observations.csv")
                    )
                    record = Table4RunResult(
                        architecture,
                        corpus,
                        seed,
                        run.info.run_id,
                        100 * result.semantic.require_score(),
                        100 * result.syntactic.require_score(),
                        100 * result.overall.require_score(),
                        result.overall.valid_count,
                        result.overall.total_count,
                        seconds,
                    )
                    evaluation_cache.store(record)
                    records.append(record)
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    unavailable.append(
                        {
                            "run_id": run.info.run_id,
                            "status": "no_data",
                            "reason": str(exc),
                        }
                    )
        (output / "unavailable.json").write_text(
            json.dumps(unavailable, indent=2) + "\n"
        )
        print(
            f"W2V1 {corpus} Table 4 analogy cache: {cache_hits} hit(s), {cache_misses} miss(es)",
            file=sys.stderr,
        )
        with (output / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=Table4RunResult.__dataclass_fields__
            )
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
        _write_summary(output / "summary.md", corpus, records, question_digest)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "protocol": {
                        "questions_sha256": question_digest,
                        "vocabulary_limit": None,
                        "analogy": "b - a + c; cosine; exclude a, b, c",
                    },
                    "reference": _REFERENCE,
                    "runs": [asdict(record) for record in records],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs.append(output)
    return tuple(outputs)


def _write_summary(
    path: Path, corpus: str, records: list[Table4RunResult], digest: str
) -> None:
    lines = [
        f"# W2V1 Table 4 — {corpus.upper()}",
        "",
        "Only conditions with all three durable canonical runs are shown.",
        "",
        "| Architecture | Corpus | Runs | Semantic mean ± sample SD (%) | Syntactic mean ± sample SD (%) | Total mean ± sample SD (%) | Questions included / total | Observed training time mean ± sample SD (s) | MLflow run IDs | Paper reference: semantic / syntactic / total (%) |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for architecture in _REFERENCE:
        values = [record for record in records if record.architecture == architecture]
        if not values:
            if architecture.startswith("nnlm-") and corpus == "fineweb":
                reference = " / ".join(
                    f"{number:.1f}" for number in _REFERENCE[architecture]
                )
                lines.append(
                    f"| {architecture} | {corpus} | 0 | unavailable | unavailable | unavailable | — | — | awaiting canonical runs | {reference} |"
                )
            continue
        coverage = ", ".join(
            f"{x.included_questions}/{x.total_questions}" for x in values
        )
        reference = " / ".join(f"{number:.1f}" for number in _REFERENCE[architecture])
        lines.append(
            f"| {architecture} | {corpus} | {len(values)} | "
            f"{_summary(values, 'semantic_accuracy_percent')} | {_summary(values, 'syntactic_accuracy_percent')} | "
            f"{_summary(values, 'total_accuracy_percent')} | {coverage} | "
            f"{_summary(values, 'observed_training_seconds')} | "
            f"{', '.join(x.mlflow_run_id for x in values)} | {reference} |"
        )
    lines.extend(
        [
            "",
            "External published vectors — paper reference only; no reproduction runs:",
            "",
            "| Model | Dimensions | Words | Semantic | Syntactic | Total (%) |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {name} | {dim} | {words} | {sem} | {synt} | {total} |"
        for name, dim, words, sem, synt, total in _EXTERNAL_REFERENCE
    )
    lines.extend(
        [
            "",
            "Our NNLM 20/50/100d uses the existing FineWeb 2013 6B surrogate policy. History=8/hidden=640/tanh and concrete PS/AdaGrad schedule are reconstruction decisions, not Table 4/6 specifications.",
            "Table 6 reuses the canonical Our NNLM 100d run IDs; it adds no NNLM training run.",
            "Paper reference: https://arxiv.org/pdf/1301.3781 (Tables 4 and 6).",
            "The 1M lexical vocabulary cap is a reconstruction decision; Table 4 does not state it explicitly.",
            "Evaluation uses the full saved lookup vocabulary. OOV questions are excluded.",
            "Scores preserve the existing compute-accuracy casing, cosine normalization, positive-score, and source-exclusion behavior.",
            f"questions-words.txt SHA-256: `{digest}`.",
            "Observed time sums final dense elapsed observations per epoch and may omit each epoch's tail.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(values: list[Table4RunResult], field: str) -> str:
    numbers = [float(getattr(value, field)) for value in values]
    return f"{statistics.fmean(numbers):.2f} ± {statistics.stdev(numbers):.2f}"
