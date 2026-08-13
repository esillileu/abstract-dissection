"""Phase 2 selector for the production fused CUDA float32 TimeLSTM."""

from mlprosection.nn.layers import TimeLSTM


Phase2TimeLSTM = TimeLSTM


def replace_better_rnnlm_lstms(_model) -> None:
    """Phase 2 is now the production TimeLSTM implementation."""
