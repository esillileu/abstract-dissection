"""Phase 3 selectors for temporal-softmax before/after comparisons."""

from deepscratch.nn.objective.classification import TemporalSoftmaxCrossEntropy


class UnfusedTemporalSoftmaxCrossEntropy(TemporalSoftmaxCrossEntropy):
    """Production objective with the Phase 3 CUDA fast path disabled."""

    _fused_cuda = False


Phase3TemporalSoftmaxCrossEntropy = TemporalSoftmaxCrossEntropy
