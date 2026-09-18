from __future__ import annotations

import csv
import re
from pathlib import Path

from .constants import ATOMIC_RUN_IDS, CSV_FIELDS, ORIGINAL_NATIVE_IDS, TOP_K
from .evaluation import RankedCandidate, RunEvaluation


def _candidate_text(candidates: tuple[RankedCandidate, ...]) -> str:
    return ", ".join(
        f"{candidate.word} ({candidate.score:.6f})" for candidate in candidates
    )


def _series_label(series: str, variant: str | None) -> str:
    if variant == "original" and series in ORIGINAL_NATIVE_IDS:
        return ORIGINAL_NATIVE_IDS[series]
    return series


def _text(
    evaluations: list[RunEvaluation],
    missing: dict[str, int],
    *,
    variant: str | None = None,
    atomic_run_ids: tuple[str, ...] = ATOMIC_RUN_IDS,
) -> str:
    lines = ["e02 PTB Word2Vec embedding evaluation (book queries; top 5)"]
    for series in atomic_run_ids:
        label = _series_label(series, variant)
        selected = [item for item in evaluations if item.series == series]
        if not selected:
            lines.append(f"[{label}] no completed runs with readable final checkpoints")
            continue
        for evaluation in selected:
            lines.append(
                f"[{label}] seed={evaluation.seed}, run_id={evaluation.run_id}"
            )
            for result in evaluation.similarities:
                lines.append(
                    f"similarity {result.query}: {_candidate_text(result.candidates)}"
                )
            for result in evaluation.analogies:
                expected_rank = (
                    "n/a" if result.expected_rank is None else str(result.expected_rank)
                )
                lines.append(
                    f"analogy {result.query} expected={result.expected}, "
                    f"rank={expected_rank}, hit@5={'yes' if result.hit_at_5 else 'no'}: "
                    f"{_candidate_text(result.candidates)}"
                )
        if missing.get(series):
            lines.append(f"[{label}] skipped unreadable checkpoints: {missing[series]}")
    return "\n".join(lines) + "\n"


def _csv_rows(evaluations: list[RunEvaluation]):
    for evaluation in evaluations:
        base = {
            "series": evaluation.series,
            "seed": evaluation.seed,
            "run_id": evaluation.run_id,
        }
        for result in evaluation.similarities:
            for rank, candidate in enumerate(result.candidates, start=1):
                yield {
                    **base,
                    "task": "similarity",
                    "query": result.query,
                    "expected": "",
                    "expected_rank": "",
                    "hit_at_5": "",
                    "candidate_rank": rank,
                    "candidate": candidate.word,
                    "score": f"{candidate.score:.9g}",
                }
        for result in evaluation.analogies:
            for rank, candidate in enumerate(result.candidates, start=1):
                yield {
                    **base,
                    "task": "analogy",
                    "query": result.query,
                    "expected": result.expected,
                    "expected_rank": (
                        "" if result.expected_rank is None else result.expected_rank
                    ),
                    "hit_at_5": "true" if result.hit_at_5 else "false",
                    "candidate_rank": rank,
                    "candidate": candidate.word,
                    "score": f"{candidate.score:.9g}",
                }


def _output_paths(output: Path) -> tuple[Path, Path]:
    stem = output.stem
    for error_style in ("band", "errorbar"):
        marker = f"_{error_style}"
        if marker in stem:
            stem = stem.replace(marker, "_word_vectors", 1)
            break
    else:
        stem = f"{stem}_word_vectors"
    return output.with_name(stem).with_suffix(".txt"), output.with_name(
        stem
    ).with_suffix(".csv")


def _write_csv(path: Path, evaluations: list[RunEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(evaluations))


def append_markdown_report(
    summary_path: Path,
    text_path: Path,
    *,
    seed: int | None = None,
) -> None:
    """Append a score-free ranked candidate table below the scalar summary."""
    summary = summary_path.read_text(encoding="utf-8").rstrip()
    summary = summary.split("\n## Word2Vec embedding evaluation", 1)[0].rstrip()
    report = _markdown_tables(
        text_path.read_text(encoding="utf-8"),
        selected_seed="1" if seed is None else str(seed),
    )
    summary_path.write_text(
        summary + "\n\n## Word2Vec embedding evaluation\n\n" + report + "\n",
        encoding="utf-8",
    )


def _markdown_tables(text: str, *, selected_seed: str | None = None) -> str:
    """Convert the renderer's detailed text into ranked Markdown tables."""
    lines = text.splitlines()
    groups: list[tuple[str, str, str, list[tuple[str, str, list[str]]]]] = []
    current_series = current_seed = current_run_id = None
    current_rows: list[tuple[str, str, list[str]]] = []

    def flush() -> None:
        if current_series is not None and current_seed is not None:
            groups.append(
                (
                    current_series,
                    current_seed,
                    current_run_id or "",
                    current_rows.copy(),
                )
            )

    for line in lines:
        header = re.match(r"^\[(.+)\] seed=(.+?), run_id=(.+)$", line)
        if header:
            flush()
            current_series, current_seed, current_run_id = header.groups()
            current_rows = []
            if selected_seed is not None and current_seed != selected_seed:
                current_series = None
            continue
        similarity = re.match(r"^similarity ([^:]+): (.+)$", line)
        analogy = re.match(
            r"^analogy (.+?) expected=([^,]+), rank=([^,]+), "
            r"hit@5=([^:]+): (.+)$",
            line,
        )
        if current_series is None or (similarity is None and analogy is None):
            continue
        if similarity is not None:
            task = "similarity"
            question, candidates = similarity.groups()
        else:
            task = "analogy"
            question, expected, rank, hit, candidates = analogy.groups()
            question = f"{question} (expected={expected}, rank={rank}, hit@5={hit})"
        words = [item.rsplit(" (", 1)[0] for item in candidates.split(", ")]
        current_rows.append((task, question, words[:TOP_K]))
    flush()

    if not groups:
        return "No completed runs with readable final checkpoints."

    output: list[str] = []
    last_series = None
    for series, seed, run_id, rows in groups:
        if series != last_series:
            if output:
                output.append("")
            output.extend((f"### {series}", ""))
            last_series = series
        output.extend((f"#### seed {seed} (`{run_id}`)", ""))
        for task, title in (("similarity", "Similarity"), ("analogy", "Analogy")):
            task_rows = [row for row in rows if row[0] == task]
            if not task_rows:
                continue
            output.extend(
                (
                    f"**{title}**",
                    "",
                    "| question | 1위 | 2위 | 3위 | 4위 | 5위 |",
                    "| --- | --- | --- | --- | --- | --- |",
                )
            )
            for _, question, words in task_rows:
                output.append(
                    "| "
                    + " | ".join(
                        [question, *words, *("" for _ in range(TOP_K - len(words)))]
                    )
                    + " |"
                )
            output.append("")
    return "\n".join(output).rstrip()
