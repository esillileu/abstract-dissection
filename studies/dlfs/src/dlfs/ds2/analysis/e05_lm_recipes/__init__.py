"""DS2 GT05: compare validation perplexity on a broken y-axis.

Split across:
* ``constants`` - Definitions, panel height ratios, limits
* ``curves``    - Curve aggregation and evaluation loading
* ``renderer``  - Plot rendering and figure saving routines
"""

from __future__ import annotations

from repro_core.analysis.core import save_figure

from .constants import (
    ADDITIONAL_BETTER_GRAPH,
    ADDITIONAL_BETTER_VALIDATION_GRAPH,
    ADDITIONAL_LSTM_GRAPH,
    ADDITIONAL_RNNLM_GRAPH,
    DEFINITIONS,
    PANEL_HEIGHT_RATIOS,
    UPPER_LOG_LINEAR_THRESHOLD,
    UPPER_Y_LIMITS,
)
from .curves import (
    _train_curves,
    _valid_curves,
)
from .renderer import (
    render,
    render_additional_better_graph,
    render_additional_better_validation_graph,
    render_additional_graph,
    render_additional_lstm_graph,
)

__all__ = [
    "ADDITIONAL_BETTER_GRAPH",
    "ADDITIONAL_BETTER_VALIDATION_GRAPH",
    "ADDITIONAL_LSTM_GRAPH",
    "ADDITIONAL_RNNLM_GRAPH",
    "DEFINITIONS",
    "PANEL_HEIGHT_RATIOS",
    "UPPER_LOG_LINEAR_THRESHOLD",
    "UPPER_Y_LIMITS",
    "_train_curves",
    "_valid_curves",
    "render",
    "render_additional_better_graph",
    "render_additional_better_validation_graph",
    "render_additional_graph",
    "render_additional_lstm_graph",
    "save_figure",
]
