"""DS2's language/sequence projection of trainer events to raw records."""

from __future__ import annotations

from ._csv import (
    append_csv,
    csv_value,
    materialize_scalars,
    source_curve_metric_name,
)
from ._records import DS2Records

# Preserve the module-level name that external code may import directly.
_source_curve_metric_name = source_curve_metric_name

__all__ = [
    "DS2Records",
    "_source_curve_metric_name",
    "append_csv",
    "csv_value",
    "materialize_scalars",
    "source_curve_metric_name",
]
