"""Constants and visual configuration for DS2 GT05 language model recipes."""

from repro_core.plotting.theme import ACCENT_COLORS

DEFINITIONS = [
    ("LM-RNN-RECIPE", "Vanilla RNNLM", "o", ACCENT_COLORS[0]),
    ("LM-LSTM-RECIPE", "LSTM RNNLM", "s", ACCENT_COLORS[1]),
    ("LM-LSTM-TIED-RECIPE", "Tied RNNLM", "^", ACCENT_COLORS[2]),
    ("LM-BETTER-RECIPE", "Better RNNLM", "D", ACCENT_COLORS[3]),
    ("LM-BETTER-NODROPOUT", "Better RNNLM (no dropout)", "X", ACCENT_COLORS[4]),
]

# Manually choose the visible Vanilla RNNLM range here.
UPPER_Y_LIMITS = (100, 10000000)
UPPER_LOG_LINEAR_THRESHOLD = 250
# Increase the first value to move the wave break downward.
PANEL_HEIGHT_RATIOS = (2, 3)

ADDITIONAL_RNNLM_GRAPH = (
    ("LM-RNN-RECIPE", "RNN RNNLM", "o", ACCENT_COLORS[0]),
    ("LM-LSTM-RECIPE", "LSTM RNNLM", "s", ACCENT_COLORS[1]),
)
ADDITIONAL_BETTER_GRAPH = (
    ("LM-BETTER-RECIPE", "Better RNNLM", "D", ACCENT_COLORS[3]),
    ("LM-BETTER-NODROPOUT", "Better RNNLM (no dropout)", "X", ACCENT_COLORS[4]),
)
ADDITIONAL_BETTER_VALIDATION_GRAPH = (
    ("LM-BETTER-RECIPE", "Better RNNLM", "D", ACCENT_COLORS[3]),
)
ADDITIONAL_LSTM_GRAPH = (
    ("LM-LSTM-RECIPE", "RNNLM", "s", ACCENT_COLORS[1]),
    ("LM-LSTM-TIED-RECIPE", "RNNLM(weight tying)", "^", ACCENT_COLORS[2]),
)
