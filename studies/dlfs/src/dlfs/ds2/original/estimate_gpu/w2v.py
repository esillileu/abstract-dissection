"""Word2Vec CuPy runtime estimation."""

from __future__ import annotations

import importlib
import sys
import types
from time import perf_counter

import numpy as np

from .schema import (
    BOOK_ROOT,
    PTB_TRAIN,
    RuntimeEstimate,
    _add_import_path,
)


def _prepare_word2vec():
    _add_import_path(BOOK_ROOT)
    config = importlib.import_module("common.config")
    config.GPU = True
    cp = importlib.import_module("cupy")
    compatibility_module = types.ModuleType("common.np")
    compatibility_module.GPU = True
    compatibility_module.np = cp
    sys.modules["common.np"] = compatibility_module
    trainer_module = importlib.import_module("common.trainer")
    optimizer_module = importlib.import_module("common.optimizer")
    util_module = importlib.import_module("common.util")

    corpus = np.load(PTB_TRAIN)
    contexts, target = util_module.create_contexts_target(corpus, 5)
    return cp, trainer_module, optimizer_module, contexts, target, corpus


def estimate_word2vec(
    model_name: str,
    *,
    benchmark_updates: int,
) -> RuntimeEstimate:
    cp, trainer_module, optimizer_module, contexts, target, corpus = _prepare_word2vec()
    if model_name == "CBOW":
        model_class = importlib.import_module("ch04.cbow").CBOW
    elif model_name == "SkipGram":
        model_class = importlib.import_module("ch04.skip_gram").SkipGram
    else:
        raise ValueError(f"unknown Word2Vec model: {model_name}")

    model = model_class(10_000, 100, 5, corpus)
    optimizer = optimizer_module.Adam()
    batch_size = 100
    data_size = len(contexts)
    updates_per_epoch = data_size // batch_size
    total_updates = updates_per_epoch * 10

    def update(index: int):
        start = (index % updates_per_epoch) * batch_size
        batch_x = contexts[start : start + batch_size]
        batch_t = target[start : start + batch_size]
        loss = model.forward(batch_x, batch_t)
        model.backward()
        params, grads = trainer_module.remove_duplicate(model.params, model.grads)
        optimizer.update(params, grads)
        return loss

    for index in range(2):
        update(index)
    cp.cuda.get_current_stream().synchronize()

    total_loss = 0
    started = perf_counter()
    for index in range(benchmark_updates):
        total_loss += update(index + 2)
        if index % 20 == 0:
            float(total_loss)
            total_loss = 0
    cp.cuda.get_current_stream().synchronize()
    seconds_per_update = (perf_counter() - started) / benchmark_updates

    # Original Trainer performs one CPU permutation and two GPU gathers per epoch.
    shuffle_times = []
    for _ in range(2):
        indices = np.random.permutation(np.arange(data_size))
        started = perf_counter()
        shuffled_contexts = contexts[indices]
        shuffled_target = target[indices]
        cp.cuda.get_current_stream().synchronize()
        shuffle_times.append(perf_counter() - started)
        del shuffled_contexts, shuffled_target
    shuffle_time = min(shuffle_times)

    compute_time = seconds_per_update * total_updates
    overhead_time = shuffle_time * 10
    return RuntimeEstimate(
        experiment_id="e02",
        condition=f"{model_name}-NegativeSampling",
        source="ch04/train.py",
        benchmark_units=benchmark_updates,
        benchmark_unit="update",
        seconds_per_unit=seconds_per_update,
        total_units=total_updates,
        projected_compute_time_s=compute_time,
        projected_overhead_time_s=overhead_time,
        projected_total_time_s=compute_time + overhead_time,
    )
