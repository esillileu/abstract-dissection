"""W2V1 table analysis package."""

from __future__ import annotations

from .table02 import (
    CORPUS_SOURCES,
    Table2RunResult,
    _complete_conditions,
    analyze_table2,
    analyze_table2_sources,
)
from .table03 import (
    Table3RunResult,
    analyze_table3_sources,
)
from .table04 import (
    Table4RunResult,
    analyze_table4_sources,
)
from .table05 import (
    Table5RunResult,
    analyze_table5_sources,
)
from .table06 import (
    Table6RunResult,
    analyze_table6_sources,
)
from .table07 import (
    Table7RunResult,
    analyze_table7_sources,
)

__all__ = [
    "CORPUS_SOURCES",
    "Table2RunResult",
    "Table3RunResult",
    "Table4RunResult",
    "Table5RunResult",
    "Table6RunResult",
    "Table7RunResult",
    "_complete_conditions",
    "analyze_table2",
    "analyze_table2_sources",
    "analyze_table3_sources",
    "analyze_table4_sources",
    "analyze_table5_sources",
    "analyze_table6_sources",
    "analyze_table7_sources",
]
