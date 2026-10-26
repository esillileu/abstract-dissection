"""Full-vocabulary Table 3 comparison on both word-relationship benchmarks."""

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
from .resources import MSR_SYNTACTIC_URL

_SEEDS = {1, 7, 19}
_PLANS = {
    "wmt": "w2v1-reconstruction-r2",
    "lm1b": "w2v1-lm1b-reconstruction-r1",
    "umbc": "w2v1-umbc-reconstruction-r1",
}
_CONDITIONS = {
    "cbow": "cbow-d640-w320m",
    "skipgram": "skipgram-d640-w320m",
    "nnlm": "nnlm-d640-h640-n8-w320m",
}
_REFERENCE = {
    "rnnlm": (9, 36, 35),
    "nnlm": (23, 53, 47),
    "cbow": (24, 64, 61),
    "skipgram": (55, 59, 56),
}


@dataclass(frozen=True)
class Table3RunResult:
    architecture: str
    corpus: str
    seed: int
    mlflow_run_id: str
    semantic_accuracy_percent: float
    syntactic_accuracy_percent: float
    total_accuracy_percent: float
    msr_accuracy_percent: float
    semantic_included: int
    semantic_total: int
    syntactic_included: int
    syntactic_total: int
    msr_included: int
    msr_total: int
    observed_training_seconds: float


_EVALUATION_PROTOCOL = "table3-v1"


class Table3EvaluationCache:
    """Signature-checked per-run cache for reconstructible Table 3 analogy results."""

    def __init__(self, root: Path, *, questions_sha256: str, msr_sha256: str) -> None:
        self.root = root
        self.signature = {
            "schema_version": 2,
            "evaluation_protocol": _EVALUATION_PROTOCOL,
            "questions_sha256": questions_sha256,
            "msr_sha256": msr_sha256,
        }

    def load(self, run_id: str) -> Table3RunResult | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != self.signature:
                return None
            result = Table3RunResult(**payload["result"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return result if result.mlflow_run_id == run_id else None

    def store(self, result: Table3RunResult) -> None:
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
        for architecture, condition in _CONDITIONS.items():
            if tags.get("implementation.variant") != f"{corpus}--{condition}":
                continue
            if tags.get("experiment_spec.id") != f"w2v1-table3-{architecture}":
                continue
            seed = int(tags.get("seed", "0"))
            if (
                seed not in _SEEDS
                or tags.get("f2.planned_run_slot_id")
                != f"{_PLANS[corpus]}-{condition}-s{seed}"
            ):
                continue
            grouped.setdefault(architecture, {}).setdefault(seed, run)
    return {
        architecture: seeds
        for architecture, seeds in grouped.items()
        if set(seeds) == _SEEDS
    }


def analyze_table3_sources(
    cache_root: Path,
    questions_path: Path,
    *,
    msr_questions_path: Path | None = None,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:

    paths = paths or RuntimePaths.from_environment()
    if not Path(questions_path).is_file():
        output = unavailable_report(
            paths.cache_root / "f2/w2v1/analysis/table3",
            "cached evaluation questions missing",
        )
        return (output,)
    if corpus_source is not None and corpus_source not in CORPUS_SOURCES:
        raise ValueError(f"unsupported W2V1 corpus source: {corpus_source}")
    google_bytes = Path(questions_path).read_bytes()
    questions = parse_analogy_questions(google_bytes.splitlines())
    if len(questions) != 19_544:
        raise ValueError(
            f"canonical questions-words.txt must contain 19,544 questions; found {len(questions)}"
        )
    msr_path = Path(
        msr_questions_path or Path(cache_root) / "evaluation/MSR-syntax.txt"
    )
    if not msr_path.is_file():
        return (
            unavailable_report(
                paths.cache_root / "f2/w2v1/analysis/table3",
                "cached MSR questions missing",
            ),
        )
    msr_bytes = msr_path.read_bytes()
    msr = parse_analogy_questions(msr_bytes.splitlines())
    google_digest = hashlib.sha256(google_bytes).hexdigest()
    msr_digest = hashlib.sha256(msr_bytes).hexdigest()
    cache = LocalResults(cache_root)
    runs = cache.runs()

    evaluation_cache = Table3EvaluationCache(
        paths.cache_root / "f2/w2v1/table3",
        questions_sha256=google_digest,
        msr_sha256=msr_digest,
    )
    outputs = []
    sources = CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    for corpus in sources:
        output = (paths.cache_root / "f2/w2v1/analysis") / corpus / "table3"
        output.mkdir(parents=True, exist_ok=True)
        records = []
        unavailable = []
        cache_hits = 0
        cache_misses = 0
        for architecture, seeds in sorted(complete_conditions(runs, corpus).items()):
            for seed, run in sorted(seeds.items()):
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
                    lookup = load_lookup_artifact(cache.get(run.info.run_id, "lookup"))
                    word_result = evaluate_analogies(
                        lookup, questions, vocabulary_limit=None, batch_size=256
                    )
                    msr_result = evaluate_analogies(
                        lookup, msr, vocabulary_limit=None, batch_size=256
                    ).overall
                    seconds = observed_training_seconds(
                        cache.get(run.info.run_id, "metrics/observations.csv")
                    )
                    record = Table3RunResult(
                        architecture,
                        corpus,
                        seed,
                        run.info.run_id,
                        100 * word_result.semantic.score,
                        100 * word_result.syntactic.score,
                        100 * word_result.overall.score,
                        100 * msr_result.score,
                        word_result.semantic.valid_count,
                        word_result.semantic.total_count,
                        word_result.syntactic.valid_count,
                        word_result.syntactic.total_count,
                        msr_result.valid_count,
                        msr_result.total_count,
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
            f"W2V1 {corpus} Table 3 analogy cache: {cache_hits} hit(s), {cache_misses} miss(es)",
            file=sys.stderr,
        )
        with (output / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=Table3RunResult.__dataclass_fields__
            )
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
        _write_summary(
            output / "summary.md", corpus, records, google_digest, msr_digest
        )
        outputs.append(output)
    return tuple(outputs)


def _write_summary(
    path: Path,
    corpus: str,
    records: list[Table3RunResult],
    google_digest: str,
    msr_digest: str,
) -> None:
    lines = [
        f"# W2V1 Table 3 — {corpus.upper()}",
        "",
        "Only conditions with all three durable canonical seed runs are shown. Scores are mean ± sample SD (%).",
        "",
        "| Architecture | Runs | Semantic | Syntactic | MSR syntactic | Included / total: semantic; syntactic; MSR (per seed) | Observed training seconds | MLflow IDs | Paper reference: semantic / syntactic / MSR (%) |",
        "|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for architecture in _CONDITIONS:
        values = [record for record in records if record.architecture == architecture]
        if not values:
            continue
        reference = " / ".join(str(value) for value in _REFERENCE[architecture])
        coverage = ", ".join(
            f"s{x.seed}: {x.semantic_included}/{x.semantic_total}; {x.syntactic_included}/{x.syntactic_total}; {x.msr_included}/{x.msr_total}"
            for x in values
        )

        lines.append(
            f"| {architecture} | {len(values)} | {_summary(values, 'semantic_accuracy_percent')} | {_summary(values, 'syntactic_accuracy_percent')} | {_summary(values, 'msr_accuracy_percent')} | {coverage} | {_summary(values, 'observed_training_seconds')} | {', '.join(x.mlflow_run_id for x in values)} | {reference} |"
        )
    lines.extend(
        [
            "",
            "RNNLM remains a paper reference only (semantic / syntactic / MSR: 9 / 36 / 35); it is a separate follow-up.",
            "Both benchmarks use the full saved input-embedding vocabulary and the unchanged compute-accuracy protocol: float32 normalization, ASCII casing, b - a + c, positive cosine, exclude a/b/c, exclude OOV and report coverage.",
            "NNLM uses the shared input/projection embedding, not hidden or HS output weights.",
            "NNLM tanh, 3 epochs and concrete PS/AdaGrad settings are reconstruction decisions, recorded in e02_table3.yaml and existing catalog notes. Corpus substitutions are explicit.",
            "The MSR syntactic benchmark is a commit-pinned public mirror of 8,000 word analogies, distinct from Sentence Completion. Original Microsoft distribution bytes are not claimed.",
            f"questions-words.txt SHA-256: `{google_digest}`.",
            f"MSR-syntax.txt SHA-256: `{msr_digest}`.",
            f"MSR mirror: {MSR_SYNTACTIC_URL}",
            "Paper reference: https://arxiv.org/pdf/1301.3781 (Table 3).",
            "Observed training time sums the final dense elapsed observation per epoch and may omit each epoch's tail. It is separate from paper reference accuracies.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(values: list[Table3RunResult], field: str) -> str:
    numbers = [float(getattr(value, field)) for value in values]
    return f"{statistics.fmean(numbers):.2f} ± {statistics.stdev(numbers):.2f}"
