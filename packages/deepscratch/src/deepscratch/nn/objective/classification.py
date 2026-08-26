from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.nn.functional import (
    binary_cross_entropy_with_logits,
    softmax_cross_entropy,
)
from deepscratch.nn.kernels.temporal_softmax import temporal_softmax_loss_gradient

from .base import Objective, ObjectiveResult


class SoftmaxCrossEntropy(Objective):
    def __init__(
        self,
        reduction: str = "mean",
        *,
        ignore_label: int | None = None,
        backend=None,
    ) -> None:
        super().__init__(backend)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.ignore_label = ignore_label
        self._gradient: Tensor | None = None

    def forward_manual(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        cache: bool = True,
        replay_context=None,
    ) -> ObjectiveResult:
        computation = softmax_cross_entropy(
            prediction,
            target,
            reduction=self.reduction,
            ignore_label=self.ignore_label,
        )
        if cache:
            self._gradient = computation.gradient
        return ObjectiveResult(
            loss=computation.loss,
            unit_count=computation.unit_count,
        )

    def backward_manual(self) -> Tensor:
        if self._gradient is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        return self._gradient


class TemporalSoftmaxCrossEntropy(SoftmaxCrossEntropy):
    """Softmax cross entropy for ``(batch, time, classes)`` predictions."""

    _fused_cuda = True

    def __init__(
        self,
        reduction: str = "mean",
        *,
        ignore_label: int | None = -1,
        backend=None,
    ) -> None:
        super().__init__(
            reduction=reduction,
            ignore_label=ignore_label,
            backend=backend,
        )

    def forward_manual(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        cache: bool = True,
        replay_context=None,
    ) -> ObjectiveResult:
        if prediction.ndim != 3:
            raise ValueError(
                "TemporalSoftmaxCrossEntropy expects (batch, time, classes)"
            )
        xp = prediction.backend.xp
        if (
            self._fused_cuda
            and prediction.backend.is_gpu
            and prediction.dtype == xp.float32
            and target.shape != prediction.shape
        ):
            loss, gradient, unit_count = temporal_softmax_loss_gradient(
                prediction.data,
                target.data,
                reduction=self.reduction,
                ignore_label=self.ignore_label,
                backend=prediction.backend,
            )
            if cache:
                self._gradient = Tensor(gradient, backend=prediction.backend)
            return ObjectiveResult(
                loss=Tensor(loss, backend=prediction.backend),
                unit_count=unit_count,
            )
        return super().forward_manual(
            prediction,
            target,
            cache=cache,
            replay_context=replay_context,
        )


class BinaryCrossEntropyWithLogits(Objective):
    """Binary cross entropy supporting arbitrary prediction shapes."""

    def __init__(self, reduction: str = "mean", *, backend=None) -> None:
        super().__init__(backend)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self._gradient: Tensor | None = None

    def forward_manual(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        cache: bool = True,
        replay_context=None,
    ) -> ObjectiveResult:
        computation = binary_cross_entropy_with_logits(
            prediction,
            target,
            reduction=self.reduction,
        )
        if cache:
            self._gradient = computation.gradient
        return ObjectiveResult(
            loss=computation.loss,
            unit_count=computation.unit_count,
        )

    def backward_manual(self) -> Tensor:
        if self._gradient is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        return self._gradient
