"""W2V2 table-by-table analysis package."""

from __future__ import annotations

from .section04_1 import (
    Section4RunResult,
    analyze_section04_1,
    analyze_section04_1_sources,
)
from .table01 import (
    Table1RunResult,
    analyze_table1,
    analyze_table1_sources,
)
from .table02 import (
    analyze_table2,
    analyze_table2_sources,
)
from .table03 import (
    Table3RunResult,
    analyze_table3,
    analyze_table3_sources,
)
from .table04 import (
    analyze_table4,
    analyze_table4_sources,
)
from .table05 import (
    analyze_table5,
    analyze_table5_sources,
)
from .table06 import (
    analyze_table6,
    analyze_table6_sources,
)

__all__ = [
    "Section4RunResult",
    "Table1RunResult",
    "Table3RunResult",
    "analyze_section04_1",
    "analyze_section04_1_sources",
    "analyze_table1",
    "analyze_table1_sources",
    "analyze_table2",
    "analyze_table2_sources",
    "analyze_table3",
    "analyze_table3_sources",
    "analyze_table4",
    "analyze_table4_sources",
    "analyze_table5",
    "analyze_table5_sources",
    "analyze_table6",
    "analyze_table6_sources",
]
