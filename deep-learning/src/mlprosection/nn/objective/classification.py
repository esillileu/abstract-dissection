from __future__ import annotations

from mlprosection import Tensor
from mlprosection.nn.layers.criterion import SoftmaxWithLoss
from mlprosection.nn.layers.time import TimeSoftmaxWithLoss

from .base import Objective, ObjectiveResult


class SoftmaxCrossEntropy(Objective):
    def __init__(self, reduction: str = "mean", *, backend=None) -> None:
        super().__init__(backend)
        if reduction != "mean":
            raise ValueError("SoftmaxCrossEntropy currently supports reduction='mean'")
        self.reduction = reduction
        self.loss = SoftmaxWithLoss()

    def forward_manual(
        self, prediction: Tensor, target: Tensor, *, cache: bool = True,
        replay_context=None,
    ) -> ObjectiveResult:
        # The elementary criterion has no cache=False path.  Evaluation/probe
        # forwards may replace its cache because backward is not called after a
        # probe.
        loss = self.loss.forward(prediction, target)
        return ObjectiveResult(loss=loss, unit_count=len(target))

    def backward_manual(self) -> Tensor:
        return self.loss.backward()


class TemporalSoftmaxCrossEntropy(Objective):
    def __init__(self, reduction: str = "mean", *, backend=None) -> None:
        super().__init__(backend)
        if reduction != "mean":
            raise ValueError(
                "TemporalSoftmaxCrossEntropy currently supports reduction='mean'"
            )
        self.reduction = reduction
        self.loss = TimeSoftmaxWithLoss()

    def forward_manual(
        self, prediction: Tensor, target: Tensor, *, cache: bool = True,
        replay_context=None,
    ) -> ObjectiveResult:
        loss = self.loss.forward(prediction, target, cache=cache)
        unit_count = target.size
        return ObjectiveResult(loss=loss, unit_count=unit_count)

    def backward_manual(self) -> Tensor:
        return self.loss.backward()
