"""DS2 GT10: timings and seed-1 predictions for count-based embeddings."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from exp.deepscratch.analysis.input import artifact_file
from mlprosection.datasets import load_ptb

from .common import runs
from .e02_ptb_word2vec import (
    RunEvaluation,
    _checkpoint_weights_path,
    _markdown_tables,
    _text as prediction_text,
    _word_vectors,
    evaluate_vectors,
)


ATOMIC_RUN_IDS = (
    "count-ptb-ppmi",
    "count-ptb-svd",
    "count-ptb-randomized-svd",
)
TIMING_FIELDS = (
    "series", "seed", "run_id", "cooccurrence_s", "ppmi_s",
    "decomposition_s", "total_s",
)


def _timing(client, run) -> dict[str, float] | None:
    path = artifact_file(client, run, "timing.json")
    if path is None:
        path = artifact_file(client, run, "raw/timing.json")
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            key: float(payload[key])
            for key in (
                "cooccurrence_s", "ppmi_s", "decomposition_s", "total_s"
            )
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def render(client, error_style, output):
    del error_style
    grouped = runs(client, "GT10", list(ATOMIC_RUN_IDS))
    ptb = load_ptb()
    evaluations: list[RunEvaluation] = []
    timing_rows: list[dict[str, object]] = []
    missing: dict[str, int] = {}
    for series in ATOMIC_RUN_IDS:
        for run in grouped[series]:
            timing = _timing(client, run)
            if timing is not None:
                timing_rows.append({
                    "series": series,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    **timing,
                })
            checkpoint = _checkpoint_weights_path(client, run)
            if checkpoint is None:
                missing[series] = missing.get(series, 0) + 1
                continue
            try:
                evaluations.append(evaluate_vectors(
                    series,
                    run.seed,
                    run.run_id,
                    _word_vectors(checkpoint),
                    ptb["word_to_id"],
                    ptb["id_to_word"],
                ))
            except (OSError, ValueError):
                missing[series] = missing.get(series, 0) + 1

    text_path = output.with_name(f"{output.stem}_count_vectors.txt")
    timing_path = output.with_name(f"{output.stem}_timings.csv")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        _timing_text(timing_rows)
        + "\n"
        + prediction_text(
            evaluations, missing, atomic_run_ids=ATOMIC_RUN_IDS
        ),
        encoding="utf-8",
    )
    with timing_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(timing_rows)
    return [text_path, timing_path]


def _timing_text(rows: list[dict[str, object]]) -> str:
    lines = ["e12 PTB count-based embedding timings"]
    for row in rows:
        lines.append(
            f"[timing {row['series']}] seed={row['seed']}, run_id={row['run_id']}, "
            f"cooccurrence={float(row['cooccurrence_s']):.6f}s, "
            f"ppmi={float(row['ppmi_s']):.6f}s, "
            f"decomposition={float(row['decomposition_s']):.6f}s, "
            f"total={float(row['total_s']):.6f}s"
        )
    return "\n".join(lines) + "\n"


def append_markdown_report(
    summary_path: Path, text_path: Path, *, seed: int | None = None
) -> None:
    selected_seed = "1" if seed is None else str(seed)
    text = text_path.read_text(encoding="utf-8")
    timing_lines = []
    for line in text.splitlines():
        if not line.startswith("[timing ") or f"seed={selected_seed}," not in line:
            continue
        label = line.removeprefix("[timing ").split("]", 1)[0]
        values = line.split("] ", 1)[1].split(", ")
        timing_lines.append((label, *values[2:]))
    timing_table = [
        "| condition | cooccurrence | PPMI | decomposition | total |",
        "| --- | ---: | ---: | ---: | ---: |",
        *("| " + " | ".join(row) + " |" for row in timing_lines),
    ]
    summary = summary_path.read_text(encoding="utf-8").rstrip()
    summary = summary.split("\n## Count-based timing and predictions", 1)[0].rstrip()
    summary_path.write_text(
        summary
        + "\n\n## Count-based timing and predictions\n\n"
        + f"Seed {selected_seed} phase timings:\n\n"
        + "\n".join(timing_table)
        + "\n\n"
        + _markdown_tables(text, selected_seed=selected_seed)
        + "\n",
        encoding="utf-8",
    )
