"""DS1 E11: compare seed-index 0 SimpleCNN filters on one shared scale."""

from __future__ import annotations

from repro_core.analysis.core import save_figure

from .checkpoint import _checkpoint_weights_path, _conv_weights, _natural_key
from .pipeline import (
    RUN_GROUPS,
    VISUALIZATION_SEED,
    _collect,
    _visualization_runs,
    render,
    render_summary,
)
from .plotting import (
    FILTER_COLOR_LIMIT,
    SUMMARY_FIELDS,
    _panel_output,
    _render_panel,
    _shared_weight_limit,
    _write_summary,
)
from .tiling import _filter_mosaic, _square_grid, _tiled_array

__all__ = [
    "FILTER_COLOR_LIMIT",
    "RUN_GROUPS",
    "SUMMARY_FIELDS",
    "VISUALIZATION_SEED",
    "_checkpoint_weights_path",
    "_collect",
    "_conv_weights",
    "_filter_mosaic",
    "_natural_key",
    "_panel_output",
    "_render_panel",
    "_shared_weight_limit",
    "_square_grid",
    "_tiled_array",
    "_visualization_runs",
    "_write_summary",
    "render",
    "render_summary",
    "save_figure",
]
