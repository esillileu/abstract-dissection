"""Language model (RNNLM, LSTM) CuPy runtime estimation."""

from __future__ import annotations

import importlib
from time import perf_counter

import numpy as np

from .schema import (
    B2_SOURCE_ROOT,
    BOOK_ROOT,
    PTB_TRAIN,
    RuntimeEstimate,
    _add_import_path,
)


def estimate_rnnlm(*, benchmark_epochs: int) -> RuntimeEstimate:
    cp = importlib.import_module("cupy")
    _add_import_path(B2_SOURCE_ROOT)
    _add_import_path(BOOK_ROOT)
    model_class = importlib.import_module("ch05.simple_rnnlm").SimpleRnnlm
    optimizer_class = importlib.import_module("b2.common.optimizer").SGD
    trainer_class = importlib.import_module("b2.common.trainer").RnnlmTrainer

    corpus = np.load(PTB_TRAIN)[:1000]
    xs, ts = corpus[:-1], corpus[1:]
    model = model_class(int(corpus.max()) + 1, 100, 100)
    optimizer = optimizer_class(0.1)
    trainer = trainer_class(model, optimizer)

    # Warm up all forward/backward kernels on one original epoch.
    trainer.fit(xs, ts, 1, 10, 5, eval_interval=20)
    cp.cuda.get_current_stream().synchronize()
    started = perf_counter()
    trainer.fit(xs, ts, benchmark_epochs, 10, 5, eval_interval=20)
    cp.cuda.get_current_stream().synchronize()
    seconds_per_epoch = (perf_counter() - started) / benchmark_epochs
    projected = seconds_per_epoch * 100
    return RuntimeEstimate(
        experiment_id="e03",
        condition="SimpleRnnlm",
        source="ch05/train.py",
        benchmark_units=benchmark_epochs,
        benchmark_unit="epoch",
        seconds_per_unit=seconds_per_epoch,
        total_units=100,
        projected_compute_time_s=projected,
        projected_overhead_time_s=0.0,
        projected_total_time_s=projected,
    )


def _eval_perplexity_iterations(
    model,
    corpus,
    *,
    iterations: int,
    batch_size: int,
    time_size: int,
    cp,
) -> None:
    corpus_size = len(corpus)
    jump = (corpus_size - 1) // batch_size
    total_loss = 0
    for iteration in range(iterations):
        xs = cp.zeros((batch_size, time_size), dtype=cp.int32)
        ts = cp.zeros((batch_size, time_size), dtype=cp.int32)
        time_offset = iteration * time_size
        offsets = [time_offset + index * jump for index in range(batch_size)]
        for time_index in range(time_size):
            for batch_index, offset in enumerate(offsets):
                xs[batch_index, time_index] = corpus[
                    (offset + time_index) % corpus_size
                ]
                ts[batch_index, time_index] = corpus[
                    (offset + time_index + 1) % corpus_size
                ]
        total_loss += model.forward(xs, ts)
    float(cp.exp(total_loss / iterations))


def estimate_lstm_rnnlm(*, benchmark_updates: int) -> RuntimeEstimate:
    cp = importlib.import_module("cupy")
    _add_import_path(B2_SOURCE_ROOT)
    model_class = importlib.import_module("b2.common.models").Rnnlm
    optimizer_class = importlib.import_module("b2.common.optimizer").SGD
    trainer_class = importlib.import_module("b2.common.trainer").RnnlmTrainer

    train_corpus = np.load(PTB_TRAIN)
    test_corpus = np.load(B2_SOURCE_ROOT / "datasets/ptb.test.npy")
    model = model_class(10_000, 100, 100)
    optimizer = optimizer_class(20.0)
    trainer = trainer_class(model, optimizer)

    tokens_per_update = 20 * 35
    warmup = train_corpus[: tokens_per_update + 1]
    trainer.fit(
        warmup[:-1],
        warmup[1:],
        1,
        20,
        35,
        max_grad=0.25,
        eval_interval=20,
    )
    cp.cuda.get_current_stream().synchronize()

    benchmark = train_corpus[: benchmark_updates * tokens_per_update + 1]
    started = perf_counter()
    trainer.fit(
        benchmark[:-1],
        benchmark[1:],
        1,
        20,
        35,
        max_grad=0.25,
        eval_interval=20,
    )
    cp.cuda.get_current_stream().synchronize()
    seconds_per_update = (perf_counter() - started) / benchmark_updates

    model.reset_state()
    eval_iterations = 50
    started = perf_counter()
    _eval_perplexity_iterations(
        model,
        test_corpus,
        iterations=eval_iterations,
        batch_size=10,
        time_size=35,
        cp=cp,
    )
    cp.cuda.get_current_stream().synchronize()
    seconds_per_eval_iteration = (perf_counter() - started) / eval_iterations

    total_train_updates = ((len(train_corpus) - 1) // (20 * 35)) * 4
    total_eval_iterations = (len(test_corpus) - 1) // (10 * 35)
    train_time = seconds_per_update * total_train_updates
    eval_time = seconds_per_eval_iteration * total_eval_iterations
    return RuntimeEstimate(
        experiment_id="e04",
        condition="LSTM-Rnnlm",
        source="ch06/train_rnnlm.py",
        benchmark_units=benchmark_updates,
        benchmark_unit="train_update",
        seconds_per_unit=seconds_per_update,
        total_units=total_train_updates,
        projected_compute_time_s=train_time,
        projected_overhead_time_s=eval_time,
        projected_total_time_s=train_time + eval_time,
    )
