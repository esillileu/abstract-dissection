from __future__ import annotations

import numpy as np
from deepscratch.core import Tensor
from deepscratch.nn.model.architecture import (
    CBOW,
    CBOWBatchAdapter,
    FusedNegativeSamplingCBOW,
    FusedNegativeSamplingSkipGram,
    SkipGram,
    SkipGramBatchAdapter,
)
from deepscratch.nn.objective import (
    FusedNegativeSampling,
    NegativeSampling,
    SoftmaxWithLoss,
)
from deepscratch.nn.sampling import UnigramSampler
from deepscratch.optim.SGD import Adam

from ..workloads import run_fused_update, run_implemented_update
from .constants import CONTEXT_WIDTH, EMBEDDING_SIZE, NEGATIVE_SAMPLES


class ScalingWorkload:
    """One current-implementation Word2Vec condition at a synthetic vocab size."""

    def __init__(
        self,
        condition: str,
        *,
        vocab_size: int,
        contexts: np.ndarray,
        targets: np.ndarray,
        backend,
    ) -> None:
        _, model_token, *variant_tokens = condition.split("-")
        variant = "-".join(variant_tokens)
        model_name = "CBOW" if model_token == "cbow" else "SkipGram"
        objective_name = (
            "FusedNegativeSampling"
            if variant == "fused-ns"
            else "NegativeSampling"
            if variant == "ns"
            else "FullSoftmax"
        )
        model_class = (
            (
                FusedNegativeSamplingCBOW
                if model_name == "CBOW"
                else FusedNegativeSamplingSkipGram
            )
            if objective_name == "FusedNegativeSampling"
            else CBOW
            if model_name == "CBOW"
            else SkipGram
        )
        self.fused = objective_name == "FusedNegativeSampling"
        self.backend = backend
        self.contexts = Tensor(
            backend.asarray(contexts, dtype=backend.xp.int64),
            backend=backend,
        )
        self.targets = Tensor(
            backend.asarray(targets, dtype=backend.xp.int64),
            backend=backend,
        )
        self.model = model_class(vocab_size, EMBEDDING_SIZE, backend=backend)
        self.adapter = (
            CBOWBatchAdapter() if model_name == "CBOW" else SkipGramBatchAdapter()
        )
        if objective_name in {"NegativeSampling", "FusedNegativeSampling"}:
            sampler = UnigramSampler.uniform(
                vocab_size,
                backend=backend,
                algorithm=UnigramSampler.CONDITIONAL_CDF,
            )
            objective_type = FusedNegativeSampling if self.fused else NegativeSampling
            self.objective = objective_type(
                vocab_size,
                negative_samples=NEGATIVE_SAMPLES,
                reduction="mean",
                sampler=sampler,
                backend=backend,
            )
        else:
            self.objective = SoftmaxWithLoss(
                reduction="mean",
                grouped_targets=model_name == "SkipGram",
                backend=backend,
            )
        params = [
            *(
                (f"model.{name}", parameter)
                for name, parameter in self.model.named_parameters()
            ),
            *(
                (f"objective.{name}", parameter)
                for name, parameter in self.objective.named_parameters()
            ),
        ]
        self.optimizer = Adam(params, lr=0.001)

    def update(self, batch_index: int, batch_size: int) -> None:
        start = batch_index * batch_size
        batch_x = self.contexts[start : start + batch_size]
        batch_t = self.targets[start : start + batch_size]
        update = run_fused_update if self.fused else run_implemented_update
        update(
            model=self.model,
            adapter=self.adapter,
            objective=self.objective,
            optimizer=self.optimizer,
            batch_x=batch_x,
            batch_t=batch_t,
        )


def _synthetic_batches(
    vocab_size: int,
    *,
    batch_size: int,
    update_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    sample_count = batch_size * update_count
    contexts = rng.integers(
        0,
        vocab_size,
        size=(sample_count, CONTEXT_WIDTH),
        dtype=np.int64,
    )
    targets = rng.integers(
        0,
        vocab_size,
        size=(sample_count, 1),
        dtype=np.int64,
    )
    return contexts, targets


def synthetic_scaling_batches(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic synthetic inputs for a scaling point."""
    return _synthetic_batches(*args, **kwargs)
