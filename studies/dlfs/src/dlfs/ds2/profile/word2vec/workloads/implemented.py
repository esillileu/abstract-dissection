from __future__ import annotations

import numpy as np
from deepscratch.core import Tensor
from deepscratch.nn.model.architecture import (
    CBOWBatchAdapter,
    DumbCBOW,
    DumbSkipGram,
    FusedNegativeSamplingCBOW,
    FusedNegativeSamplingSkipGram,
    OneHotCBOW,
    OneHotCBOWBatchAdapter,
    OneHotSkipGram,
    OneHotSkipGramBatchAdapter,
    PairExpandedSkipGramBatchAdapter,
    SkipGramBatchAdapter,
)
from deepscratch.nn.objective import (
    FusedNegativeSampling,
    NegativeSampling,
    SoftmaxWithLoss,
)
from deepscratch.nn.sampling import UnigramSampler
from deepscratch.optim.SGD import Adam
from deepscratch.profiling import SectionRecorder

from .env import _phase


class ImplementedWord2Vec:
    def __init__(
        self,
        model_name: str,
        objective_name: str,
        corpus,
        contexts,
        targets,
        backend,
        *,
        one_hot: bool = False,
    ) -> None:
        vocab_size = int(np.max(corpus)) + 1
        model_class, adapter, grouped_targets = _implemented_components(
            model_name,
            objective_name,
            vocab_size=vocab_size,
            one_hot=one_hot,
        )
        self.backend = backend
        self.contexts = Tensor(
            backend.xp.asarray(contexts, dtype=backend.xp.int64),
            backend=backend,
        )
        self.targets = Tensor(
            backend.xp.asarray(targets, dtype=backend.xp.int64),
            backend=backend,
        )
        self.model = model_class(vocab_size, 100, backend=backend)
        self.adapter = adapter
        self.fused = objective_name == "FusedNegativeSampling"
        if objective_name in {"NegativeSampling", "FusedNegativeSampling"}:
            sampler = UnigramSampler.from_corpus(
                corpus,
                vocab_size=vocab_size,
                backend=backend,
                power=0.75,
                algorithm=UnigramSampler.CONDITIONAL_CDF,
            )
            objective_type = FusedNegativeSampling if self.fused else NegativeSampling
            self.objective = objective_type(
                vocab_size,
                negative_samples=5,
                reduction="mean",
                sampler=sampler,
                backend=backend,
            )
        else:
            self.objective = SoftmaxWithLoss(
                reduction="mean",
                grouped_targets=grouped_targets,
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

    def update(self, batch_x, batch_t, recorder: SectionRecorder | None = None):
        if self.fused:
            return run_fused_update(
                model=self.model,
                adapter=self.adapter,
                objective=self.objective,
                optimizer=self.optimizer,
                batch_x=batch_x,
                batch_t=batch_t,
                recorder=recorder,
            )
        return run_implemented_update(
            model=self.model,
            adapter=self.adapter,
            objective=self.objective,
            optimizer=self.optimizer,
            batch_x=batch_x,
            batch_t=batch_t,
            recorder=recorder,
        )


def run_implemented_update(
    *,
    model,
    adapter,
    objective,
    optimizer,
    batch_x,
    batch_t,
    recorder: SectionRecorder | None = None,
):
    """Run the implemented Word2Vec update path shared by all e02 profiles."""
    with _phase(recorder, "batch_adapter"):
        model_x, objective_t = adapter.prepare(batch_x, batch_t)
    with _phase(recorder, "objective_prepare"):
        objective_batch = objective.prepare(objective_t)
    with _phase(recorder, "model_forward"):
        prediction = model.forward(
            model_x,
            candidates=objective_batch.candidates,
        )
    with _phase(recorder, "objective_forward"):
        objective.forward(
            prediction,
            objective_batch.target,
            replay_context=objective_batch.replay_context,
            example_count=len(batch_x),
        )
    with _phase(recorder, "objective_backward"):
        gradient = objective.backward()
    with _phase(recorder, "model_backward"):
        model.backward(gradient)
    with _phase(recorder, "optimizer"):
        optimizer.update()
    with _phase(recorder, "post_update_loss"):
        post_prediction = model.forward(
            model_x,
            candidates=objective_batch.candidates,
            cache=False,
        )
        post_result = objective.forward(
            post_prediction,
            objective_batch.target,
            cache=False,
            replay_context=objective_batch.replay_context,
            example_count=len(batch_x),
        )
    return post_result.loss


def run_fused_update(
    *,
    model,
    adapter,
    objective,
    optimizer,
    batch_x,
    batch_t,
    recorder: SectionRecorder | None = None,
):
    """Run the executor-equivalent fused negative-sampling update."""
    with _phase(recorder, "batch_adapter"):
        model_x, objective_t = adapter.prepare(batch_x, batch_t)
    with _phase(recorder, "objective_prepare"):
        objective_batch = objective.prepare(objective_t)
    with _phase(recorder, "fused_forward_loss"):
        objective.forward_fused(
            model,
            model_x,
            objective_batch,
            example_count=len(batch_x),
        )
    with _phase(recorder, "fused_backward"):
        objective.backward_fused(model)
    with _phase(recorder, "optimizer"):
        optimizer.update()
    with _phase(recorder, "post_update_loss"):
        post_result = objective.forward_fused(
            model,
            model_x,
            objective_batch,
            cache=False,
            example_count=len(batch_x),
        )
    return post_result.loss


def _implemented_components(
    model_name: str,
    objective_name: str,
    *,
    vocab_size: int,
    one_hot: bool,
):
    """Mirror the current DS2 executor's Word2Vec execution path."""
    if objective_name == "FusedNegativeSampling":
        if one_hot:
            raise ValueError("fused negative sampling requires embedding input")
        model_class = (
            FusedNegativeSamplingCBOW
            if model_name == "CBOW"
            else FusedNegativeSamplingSkipGram
        )
    else:
        model_class = {
            ("CBOW", False): DumbCBOW,
            ("SkipGram", False): DumbSkipGram,
            ("CBOW", True): OneHotCBOW,
            ("SkipGram", True): OneHotSkipGram,
        }[(model_name, one_hot)]
    adapter = {
        ("CBOW", False): CBOWBatchAdapter(),
        ("SkipGram", False): (
            PairExpandedSkipGramBatchAdapter()
            if objective_name == "FullSoftmax"
            else SkipGramBatchAdapter()
        ),
        ("CBOW", True): OneHotCBOWBatchAdapter(vocab_size),
        ("SkipGram", True): OneHotSkipGramBatchAdapter(vocab_size),
    }[(model_name, one_hot)]
    return (
        model_class,
        adapter,
        model_name == "SkipGram" and one_hot and objective_name == "FullSoftmax",
    )
