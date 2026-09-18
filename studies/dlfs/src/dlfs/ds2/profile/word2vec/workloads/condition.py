from __future__ import annotations

from deepscratch.profiling import (
    SectionRecorder,
    TimingStats,
    estimate_training_time,
)

from .implemented import ImplementedWord2Vec
from .original import OriginalWord2Vec


def _build_condition(
    condition: str,
    *,
    corpus,
    contexts,
    targets,
    backend,
):
    implementation_token, model_token, *variant_tokens = condition.split("-")
    variant = "-".join(variant_tokens)
    if implementation_token not in {"original", "implemented"}:
        raise ValueError(f"unknown Word2Vec implementation: {implementation_token}")
    if model_token not in {"cbow", "skipgram"}:
        raise ValueError(f"unknown Word2Vec model: {model_token}")
    if variant not in {"ns", "fused-ns", "fs", "onehot-fs"}:
        raise ValueError(f"unknown Word2Vec profile variant: {variant}")
    if variant == "fused-ns" and implementation_token != "implemented":
        raise ValueError("fused negative sampling has no original condition")
    model_name = "CBOW" if model_token == "cbow" else "SkipGram"
    objective_name = (
        "FusedNegativeSampling"
        if variant == "fused-ns"
        else "NegativeSampling"
        if variant == "ns"
        else "FullSoftmax"
    )
    one_hot = variant == "onehot-fs"
    if implementation_token == "original":
        return (
            OriginalWord2Vec(
                model_name,
                objective_name,
                corpus,
                contexts,
                targets,
                backend,
                one_hot=one_hot,
            ),
            model_name,
            objective_name,
            "original",
        )
    return (
        ImplementedWord2Vec(
            model_name,
            objective_name,
            corpus,
            contexts,
            targets,
            backend,
            one_hot=one_hot,
        ),
        model_name,
        objective_name,
        "implemented",
    )


def build_profile_condition(*args, **kwargs):
    """Build one typed Word2Vec workload behind the public adapter API."""
    return _build_condition(*args, **kwargs)


def _batch(workload, index: int, batch_size: int):
    data_size = len(workload.contexts)
    start = (index * batch_size) % (data_size - batch_size)
    return (
        workload.contexts[start : start + batch_size],
        workload.targets[start : start + batch_size],
    )


def profile_batch(workload, index: int, batch_size: int):
    """Return one deterministic cyclic profile minibatch."""
    return _batch(workload, index, batch_size)


def _run_updates(
    workload,
    *,
    start_index: int,
    updates: int,
    batch_size: int,
    recorder: SectionRecorder | None = None,
) -> None:
    for index in range(start_index, start_index + updates):
        batch_x, batch_t = _batch(workload, index, batch_size)
        workload.update(batch_x, batch_t, recorder)


def _runtime_estimates(
    mean_ms_per_update: float,
    *,
    dataset_samples: int,
    batch_size: int,
    epochs: int,
) -> tuple[int, float, float]:
    """Return drop-last updates/epoch and update-path epoch/total estimates."""
    estimate = estimate_training_time(
        TimingStats(
            count=1,
            mean_ms=mean_ms_per_update,
            stdev_ms=0.0,
            min_ms=mean_ms_per_update,
            max_ms=mean_ms_per_update,
            p50_ms=mean_ms_per_update,
            p95_ms=mean_ms_per_update,
        ),
        dataset_samples=dataset_samples,
        batch_size=batch_size,
        epochs=epochs,
    )
    return (
        estimate.updates_per_epoch,
        estimate.mean_seconds_per_epoch,
        estimate.mean_seconds_total,
    )
