from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import resolve_backend
from deepscratch.nn.functional import binary_cross_entropy_with_logits
from deepscratch.nn.sampling import UnigramSampler

from ..base import Objective, ObjectiveResult
from .batch import Word2VecObjectiveBatch


class NegativeSampling(Objective):
    def __init__(
        self,
        vocab_size: int,
        *,
        negative_samples: int = 5,
        reduction: str = "mean",
        sampler: UnigramSampler | None = None,
        backend=None,
    ) -> None:
        resolved = resolve_backend(backend)
        super().__init__(resolved)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.negative_samples = negative_samples
        self.sampler = sampler or UnigramSampler.uniform(vocab_size, backend=resolved)
        if self.sampler.vocab_size != vocab_size:
            raise ValueError("sampler vocabulary does not match objective vocabulary")
        self._cache = None

    def forward_manual(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        cache: bool = True,
        replay_context=None,
        example_count: int | None = None,
    ) -> ObjectiveResult:
        computation = binary_cross_entropy_with_logits(
            prediction,
            target,
            reduction="sum",
        )
        candidate_count = prediction.shape[-1]
        prediction_count = prediction.size // candidate_count
        reporting_divisor = (
            prediction_count * candidate_count
            if self.reduction == "mean"
            else prediction_count
        )
        reporting_loss = computation.loss / reporting_divisor
        optimized_divisor = (
            example_count if example_count is not None else reporting_divisor
        )
        loss = computation.loss / optimized_divisor
        score_gradient = computation.gradient / optimized_divisor
        if cache:
            self._cache = score_gradient
        return ObjectiveResult(
            loss,
            prediction_count,
            replay_context,
            reporting_loss=(reporting_loss if example_count is not None else None),
        )

    def backward_manual(self) -> Tensor:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        return self._cache

    def prepare(
        self,
        target: Tensor,
        *,
        replay_context=None,
    ) -> Word2VecObjectiveBatch:
        xp = self.backend.xp
        labels = target.data.astype(xp.int64, copy=False)
        flat_labels = labels.reshape(-1)
        negatives = (
            self.sampler.sample(flat_labels, sample_size=self.negative_samples)
            if replay_context is None
            else xp.asarray(replay_context, dtype=xp.int64)
        )
        negatives = negatives.reshape((*labels.shape, self.negative_samples))
        candidates = xp.concatenate((labels[..., None], negatives), axis=-1)
        binary_targets = xp.zeros(
            candidates.shape,
            dtype=self.backend.float_dtype,
        )
        binary_targets[..., 0] = 1
        return Word2VecObjectiveBatch(
            target=Tensor(binary_targets, backend=target.backend),
            candidates=Tensor(candidates, backend=target.backend),
            replay_context=negatives.copy(),
        )


class FusedNegativeSampling(NegativeSampling):
    """Negative-sampling objective for FusedNegativeSamplingCBOW only."""

    fused = True

    def forward_fused(
        self,
        model,
        inputs: Tensor,
        batch: Word2VecObjectiveBatch,
        *,
        cache: bool = True,
        example_count: int | None = None,
    ) -> ObjectiveResult:
        if batch.candidates is None:
            raise ValueError("negative sampling requires candidate indices")
        candidate_count = batch.candidates.shape[-1]
        prediction_count = batch.candidates.size // candidate_count
        reporting_divisor = (
            prediction_count * candidate_count
            if self.reduction == "mean"
            else prediction_count
        )
        optimized_divisor = (
            example_count if example_count is not None else reporting_divisor
        )
        loss_sum = model.forward_negative_sampling(
            inputs,
            batch.candidates,
            batch.target,
            divisor=optimized_divisor,
            cache=cache,
        )
        return ObjectiveResult(
            loss_sum / optimized_divisor,
            prediction_count,
            batch.replay_context,
            reporting_loss=(
                loss_sum / reporting_divisor if example_count is not None else None
            ),
        )

    def backward_fused(self, model) -> None:
        model.backward_negative_sampling()
