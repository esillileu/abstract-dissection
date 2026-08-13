"""Estimate DS2 e02-e04 runtimes through the book's official CuPy path."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


BOOK_ROOT = Path(
    "01_deep-learning-from-base/deep-learning-from-scratch-2"
).resolve()
B2_SOURCE_ROOT = Path("01_deep-learning-from-base/src").resolve()
PTB_TRAIN = B2_SOURCE_ROOT / "datasets/ptb.train.npy"
DEFAULT_OUTPUT = Path("exp/deepscratch/ds2/original/legacy_results/fixed_seed/gpu_estimate.json")


@dataclass(frozen=True)
class RuntimeEstimate:
    experiment_id: str
    condition: str
    source: str
    benchmark_units: int
    benchmark_unit: str
    seconds_per_unit: float
    total_units: int
    projected_compute_time_s: float
    projected_overhead_time_s: float
    projected_total_time_s: float


def estimate_word2vec(
    model_name: str,
    *,
    benchmark_updates: int,
) -> RuntimeEstimate:
    cp, trainer_module, optimizer_module, contexts, target, corpus = (
        _prepare_word2vec()
    )
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


def _add_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word2vec-updates", type=int, default=100)
    parser.add_argument("--rnnlm-epochs", type=int, default=20)
    parser.add_argument("--lstm-updates", type=int, default=100)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("e02", "e03", "e04"),
        default=("e02", "e03"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if (
        args.word2vec_updates < 1
        or args.rnnlm_epochs < 1
        or args.lstm_updates < 1
    ):
        parser.error("benchmark sizes must be positive")

    results = []
    if "e02" in args.experiments:
        results.extend(
            (
                estimate_word2vec(
                    "CBOW",
                    benchmark_updates=args.word2vec_updates,
                ),
                estimate_word2vec(
                    "SkipGram",
                    benchmark_updates=args.word2vec_updates,
                ),
            )
        )
    if "e03" in args.experiments:
        results.append(estimate_rnnlm(benchmark_epochs=args.rnnlm_epochs))
    if "e04" in args.experiments:
        results.append(
            estimate_lstm_rnnlm(benchmark_updates=args.lstm_updates)
        )
    payload = {
        "method": (
            "Original deep-learning-from-scratch-2 models/trainers using their "
            "official config.GPU=True CuPy path, with common.np bypassed only "
            "because its legacy np.add.at assignment is read-only in CuPy 14. "
            "Native CuPy add.at has the same scatter-add semantics. When e02 is "
            "selected, adapted full-softmax trials are excluded."
        ),
        "book_root": str(BOOK_ROOT),
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for result in results:
        print(
            f"{result.experiment_id}/{result.condition}: "
            f"projected={result.projected_total_time_s:.1f}s",
            flush=True,
        )
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
