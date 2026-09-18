"""Markdown rendering and console formatting for analysis summaries."""

from __future__ import annotations

from pathlib import Path

from .formatting import _coordinate, _formatted_summary


def _markdown(rows: list[dict[str, object]]) -> str:
    return _markdown_summary(rows) + "\n" + _markdown_table(rows)


def _markdown_summary(rows: list[dict[str, object]]) -> str:
    lines = ["# Analysis summary", ""]
    current = None
    for row in rows:
        coordinate = _coordinate(row)
        if coordinate != current:
            if current is not None:
                lines.append("")
            lines.extend((f"## {' / '.join(str(item) for item in coordinate)}", ""))
            current = coordinate
        lines.append(f"- {_formatted_summary(row)}")
    return "\n".join(lines).rstrip() + "\n"


def _markdown_table(rows: list[dict[str, object]]) -> str:
    columns = (
        "condition",
        "variant",
        "metric",
        "unit",
        "mean",
        "sample stddev",
        "variance",
        "min",
        "max",
        "seeds",
        "unavailable reason",
    )
    lines = [
        "## Detailed statistics",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = (
            row["canonical_condition_id"],
            row["variant"],
            row["metric_id"],
            row["unit"],
            row["mean"],
            row["sample_standard_deviation"],
            row["variance"],
            row["minimum"],
            row["maximum"],
            row["seed_runs"],
            row["unavailable_reason"],
        )
        lines.append(
            "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"
        )
    return "\n".join(lines) + "\n"


def print_summary_file(path: Path) -> None:
    summary, _separator, _table = path.read_text(encoding="utf-8").partition(
        "## Detailed statistics"
    )
    print(summary.rstrip() + "\n")


def _print_rows(rows: list[dict[str, object]]) -> None:
    current = None
    for row in rows:
        coordinate = _coordinate(row)
        if coordinate != current:
            print(f"[{coordinate[0]}/{coordinate[1]}/{coordinate[2]}]")
            current = coordinate
        print(_formatted_summary(row))
