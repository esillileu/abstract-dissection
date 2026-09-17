"""Scalar per-condition summaries over the canonical selected run set."""

from .extraction import (
    _curve_summary_fallback,
    _metric_values,
    _original_ds1_accuracy_fallback,
    _parameter_count,
)
from .formatting import (
    FIELDS,
    _coordinate,
    _format_e14_number,
    _format_number,
    _formatted_summary,
    _summary_row,
)
from .rendering import (
    _markdown,
    _markdown_summary,
    _markdown_table,
    _print_rows,
    print_summary_file,
)
from .writer import (
    TRAINING_TIME,
    summary_declarations,
    write_study_summary,
)

__all__ = [
    "FIELDS",
    "TRAINING_TIME",
    "_coordinate",
    "_curve_summary_fallback",
    "_format_e14_number",
    "_format_number",
    "_formatted_summary",
    "_markdown",
    "_markdown_summary",
    "_markdown_table",
    "_metric_values",
    "_original_ds1_accuracy_fallback",
    "_parameter_count",
    "_print_rows",
    "_summary_row",
    "print_summary_file",
    "summary_declarations",
    "write_study_summary",
]
