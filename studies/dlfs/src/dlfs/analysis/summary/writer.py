"""Study analysis summary generation and file writing."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from dlfs.analysis.declarations import MetricDeclaration
from dlfs.analysis.input import StudyAnalysisInput
from dlfs.analysis.paths import result_stem
from dlfs.identity import Variant, Volume

from .extraction import _metric_values, _parameter_count
from .formatting import FIELDS, _summary_row
from .rendering import _markdown, _markdown_summary

TRAINING_TIME = MetricDeclaration(
    "training_time_s",
    "seconds",
    "train",
    "run",
    ("runtime/train_total_s",),
    ("runtime/train_total_s",),
    protocols=("book-source-v1", "legacy"),
)


def summary_declarations(
    study_id: str,
    declared: Mapping[str, tuple[MetricDeclaration, ...]],
) -> tuple[MetricDeclaration, ...]:
    return (*declared.get(study_id, ()), TRAINING_TIME)


def write_study_summary(
    data: StudyAnalysisInput,
    *,
    volume: Volume,
    study_id: str,
    metrics: Iterable[MetricDeclaration],
    output_dir: Path,
    output_variants: tuple[Variant, ...],
    print_console: bool,
    filename_suffix: str = "",
    cache_dir: Path | None = None,
) -> Path:
    path = output_dir / (
        f"{result_stem(volume, study_id, output_variants)}{filename_suffix}.md"
    )
    rows: list[dict[str, object]] = []
    conditions = data.runs(
        tuple(condition.canonical_id for condition in data.declaration.conditions)
    )
    for condition_id, runs in conditions.items():
        for metric in metrics:
            values = _metric_values(data, study_id, runs, metric)
            rows.append(
                _summary_row(study_id, condition_id, data.variant, metric, runs, values)
            )
        parameter_values = [
            value for run in runs if (value := _parameter_count(data, run)) is not None
        ]
        rows.append(
            _summary_row(
                study_id,
                condition_id,
                data.variant,
                MetricDeclaration(
                    "parameter_count",
                    "parameters",
                    "model",
                    "run",
                    (),
                    (),
                ),
                runs,
                parameter_values,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown(rows), encoding="utf-8")
    if cache_dir is not None:
        cache_path = cache_dir / (
            f"{result_stem(volume, study_id, output_variants)}"
            f"{filename_suffix}_summary.csv"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    if print_console:
        print(_markdown_summary(rows), end="")
    return path
