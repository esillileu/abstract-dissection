"""Profile and extrapolate original and implemented DS2 e02 Word2Vec runtimes.

The benchmark keeps the PTB workload, batch size, embedding size, window size,
and optimizer aligned with e02.  It runs on either NumPy/CPU or CuPy/CUDA and
reports measured update throughput plus estimated epoch and full-run times.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import json
import platform
from pathlib import Path
import sys
import types
from typing import Iterator

import numpy as np

from mlprosection import Tensor
from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.nn.model.architecture import (
    CBOW,
    CBOWBatchAdapter,
    SkipGram,
    SkipGramBatchAdapter,
)
from mlprosection.nn.objective import NegativeSampling, SoftmaxWithLoss
from mlprosection.nn.sampling import UnigramSampler
from mlprosection.optim.SGD import Adam
from mlprosection.profiling import (
    BenchmarkRunner,
    SectionRecorder,
    TimingStats,
    estimate_training_time,
)
from exp.ds2.original.run.e02 import build_full_softmax_model


ROOT = Path(__file__).resolve().parents[4]
BOOK_ROOT = ROOT / "01_deep-learning-from-base/deep-learning-from-scratch-2"
PTB_TRAIN = ROOT / "01_deep-learning-from-base/src/datasets/ptb.train.npy"
DEFAULT_OUTPUT = ROOT / "exp/ds2/profile/e02/results/update.json"
DEFAULT_EPOCHS = 10

CONDITIONS = (
    "original-cbow-ns",
    "original-cbow-fs",
    "original-skipgram-ns",
    "original-skipgram-fs",
    "implemented-cbow-ns",
    "implemented-skipgram-ns",
    "implemented-cbow-fs",
    "implemented-skipgram-fs",
)

STAGES = {
    # Always measures one cold update before the requested steady-state work.
    "update": {
        "warmup_updates": 5,
        "measured_updates": 10,
        "phase_updates": 0,
        "repetitions": 3,
    },
    # Enough samples to expose update-level variance without an epoch run.
    "estimate": {
        "warmup_updates": 20,
        "measured_updates": 50,
        "phase_updates": 0,
        "repetitions": 5,
    },
    # Long throughput windows plus a separately synchronized phase pass.
    "detail": {
        "warmup_updates": 50,
        "measured_updates": 200,
        "phase_updates": 5,
        "repetitions": 5,
    },
}


@dataclass(frozen=True)
class ConditionResult:
    condition: str
    implementation: str
    model: str
    objective: str
    batch_size: int
    dataset_samples: int
    updates_per_epoch: int
    estimated_epochs: int
    warmup_updates: int
    measured_updates: int
    repetitions: int
    cold_ms_per_update: float
    warmup_total_ms: float
    warmup_mean_ms: float
    steady_event_mean_ms_per_update: float
    steady_event_stdev_ms_per_update: float
    steady_event_min_ms_per_update: float
    steady_event_max_ms_per_update: float
    steady_event_p50_ms_per_update: float
    steady_event_p95_ms_per_update: float
    mean_ms_per_update: float
    stdev_ms_per_update: float
    min_ms_per_update: float
    max_ms_per_update: float
    samples_per_second: float
    estimated_first_epoch_seconds: float
    estimated_seconds_per_epoch: float
    estimated_repeat_stdev_seconds_per_epoch: float
    estimated_seconds_total: float
    estimated_repeat_stdev_seconds_total: float
    phase_ms_per_update: dict[str, float]
    phase_stats: dict[str, dict[str, float | int]]
    phase_share: dict[str, float]


class OriginalWord2Vec:
    def __init__(
        self,
        model_name: str,
        objective_name: str,
        corpus,
        contexts,
        targets,
        backend,
    ) -> None:
        trainer_module = importlib.import_module("common.trainer")
        optimizer_class = importlib.import_module("common.optimizer").Adam
        self.backend = backend
        self.contexts = backend.xp.asarray(contexts)
        self.targets = backend.xp.asarray(targets)
        vocab_size = int(np.max(corpus)) + 1
        if objective_name == "FullSoftmax":
            kind = "cbow" if model_name == "CBOW" else "skipgram"
            self.model = build_full_softmax_model(kind, vocab_size, 100, 5)
        else:
            module_name = (
                "ch04.cbow" if model_name == "CBOW" else "ch04.skip_gram"
            )
            model_class = getattr(
                importlib.import_module(module_name),
                model_name,
            )
            self.model = model_class(vocab_size, 100, 5, corpus)
        self.optimizer = optimizer_class()
        self.remove_duplicate = trainer_module.remove_duplicate

    def update(self, batch_x, batch_t, recorder: SectionRecorder | None = None):
        with _phase(recorder, "forward"):
            loss = self.model.forward(batch_x, batch_t)
        with _phase(recorder, "backward"):
            self.model.backward()
        with _phase(recorder, "deduplicate_shared_parameters"):
            params, grads = self.remove_duplicate(self.model.params, self.model.grads)
        with _phase(recorder, "optimizer"):
            self.optimizer.update(params, grads)
        return loss


class ImplementedWord2Vec:
    def __init__(
        self,
        model_name: str,
        objective_name: str,
        corpus,
        contexts,
        targets,
        backend,
    ) -> None:
        model_class, adapter, grouped_targets = _implemented_components(
            model_name,
            objective_name,
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
        vocab_size = int(np.max(corpus)) + 1
        self.model = model_class(vocab_size, 100, backend=backend)
        self.adapter = adapter
        if objective_name == "NegativeSampling":
            sampler = UnigramSampler.from_corpus(
                corpus,
                vocab_size=vocab_size,
                backend=backend,
                power=0.75,
                algorithm=UnigramSampler.CONDITIONAL_CDF,
            )
            self.objective = NegativeSampling(
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


def _implemented_components(model_name: str, objective_name: str):
    """Mirror the current DS2 executor's Word2Vec execution path."""
    model_class = {
        "CBOW": CBOW,
        "SkipGram": SkipGram,
    }[model_name]
    adapter = (
        CBOWBatchAdapter()
        if model_name == "CBOW"
        else SkipGramBatchAdapter()
    )
    return (
        model_class,
        adapter,
        (model_name == "SkipGram" and objective_name == "FullSoftmax"),
    )


@contextmanager
def _phase(recorder: SectionRecorder | None, name: str) -> Iterator[None]:
    if recorder is None:
        yield
    else:
        with recorder.section(name):
            yield


def _install_original_imports(device: str):
    # The book uses ``from common.np import *`` at import time.  Clear its
    # modules before switching CPU/CUDA in one default profiling process.
    for module_name in tuple(sys.modules):
        if (
            module_name == "common"
            or module_name.startswith("common.")
            or module_name == "ch04"
            or module_name.startswith("ch04.")
        ):
            sys.modules.pop(module_name, None)
    book_path = str(BOOK_ROOT)
    if book_path not in sys.path:
        sys.path.insert(0, book_path)
    config = importlib.import_module("common.config")
    use_gpu = device.startswith("cuda:")
    config.GPU = use_gpu
    xp = importlib.import_module("cupy" if use_gpu else "numpy")
    compatibility = types.ModuleType("common.np")
    compatibility.GPU = use_gpu
    compatibility.np = xp
    sys.modules["common.np"] = compatibility
    return xp


def _load_data(device: str):
    _install_original_imports(device)
    util = importlib.import_module("common.util")
    corpus = np.load(PTB_TRAIN)
    contexts, targets = util.create_contexts_target(corpus, 5)
    backend = make_backend(
        BackendConfig(
            device=device,
            dtype="float32",
            seed=1,
            profile=device.startswith("cuda:"),
        )
    )
    return backend, corpus, contexts, targets


def _build_condition(
    condition: str,
    *,
    corpus,
    contexts,
    targets,
    backend,
):
    _, model_token, objective_token = condition.split("-")
    model_name = "CBOW" if model_token == "cbow" else "SkipGram"
    objective_name = "NegativeSampling" if objective_token == "ns" else "FullSoftmax"
    if condition.startswith("original-"):
        return (
            OriginalWord2Vec(
                model_name,
                objective_name,
                corpus,
                contexts,
                targets,
                backend,
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
        ),
        model_name,
        objective_name,
        "implemented",
    )


def _batch(workload, index: int, batch_size: int):
    data_size = len(workload.contexts)
    start = (index * batch_size) % (data_size - batch_size)
    return (
        workload.contexts[start : start + batch_size],
        workload.targets[start : start + batch_size],
    )


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


def profile_condition(
    condition: str,
    *,
    corpus,
    contexts,
    targets,
    backend,
    batch_size: int,
    epochs: int,
    warmup_updates: int,
    measured_updates: int,
    phase_updates: int,
    repetitions: int,
) -> ConditionResult:
    workload, model_name, objective_name, implementation = _build_condition(
        condition,
        corpus=corpus,
        contexts=contexts,
        targets=targets,
        backend=backend,
    )
    next_index = 0

    def update_once() -> None:
        nonlocal next_index
        batch_x, batch_t = _batch(workload, next_index, batch_size)
        workload.update(batch_x, batch_t)
        next_index += 1

    benchmark = BenchmarkRunner(backend).measure_update_protocol(
        f"{condition}.update",
        update_once,
        warmup_iterations=warmup_updates,
        measured_iterations=measured_updates,
        repetitions=repetitions,
    )

    recorder = SectionRecorder(backend)
    if phase_updates:
        _run_updates(
            workload,
            start_index=next_index,
            updates=phase_updates,
            batch_size=batch_size,
            recorder=recorder,
        )
    phase_timings = recorder.stats()
    phase_means = {name: timing.mean_ms for name, timing in phase_timings.items()}
    phase_total = sum(phase_means.values())
    estimate = estimate_training_time(
        benchmark.timing,
        dataset_samples=len(contexts),
        batch_size=batch_size,
        epochs=epochs,
        cold_update_ms=benchmark.cold_ms,
    )
    first_epoch_seconds = (
        benchmark.cold_ms
        + benchmark.timing.mean_ms * max(estimate.updates_per_epoch - 1, 0)
    ) / 1_000
    return ConditionResult(
        condition=condition,
        implementation=implementation,
        model=model_name,
        objective=objective_name,
        batch_size=batch_size,
        dataset_samples=len(contexts),
        updates_per_epoch=estimate.updates_per_epoch,
        estimated_epochs=epochs,
        warmup_updates=warmup_updates,
        measured_updates=measured_updates,
        repetitions=repetitions,
        cold_ms_per_update=benchmark.cold_ms,
        warmup_total_ms=benchmark.warmup_total_ms,
        warmup_mean_ms=benchmark.warmup_mean_ms,
        steady_event_mean_ms_per_update=benchmark.event_timing.mean_ms,
        steady_event_stdev_ms_per_update=benchmark.event_timing.stdev_ms,
        steady_event_min_ms_per_update=benchmark.event_timing.min_ms,
        steady_event_max_ms_per_update=benchmark.event_timing.max_ms,
        steady_event_p50_ms_per_update=benchmark.event_timing.p50_ms,
        steady_event_p95_ms_per_update=benchmark.event_timing.p95_ms,
        mean_ms_per_update=benchmark.timing.mean_ms,
        stdev_ms_per_update=benchmark.timing.stdev_ms,
        min_ms_per_update=benchmark.timing.min_ms,
        max_ms_per_update=benchmark.timing.max_ms,
        samples_per_second=(batch_size / (benchmark.timing.mean_ms / 1_000)),
        estimated_first_epoch_seconds=first_epoch_seconds,
        estimated_seconds_per_epoch=estimate.mean_seconds_per_epoch,
        estimated_repeat_stdev_seconds_per_epoch=(
            estimate.repeat_stdev_seconds_per_epoch
        ),
        estimated_seconds_total=estimate.mean_seconds_total,
        estimated_repeat_stdev_seconds_total=estimate.repeat_stdev_seconds_total,
        phase_ms_per_update=phase_means,
        phase_stats={name: asdict(timing) for name, timing in phase_timings.items()},
        phase_share={name: value / phase_total for name, value in phase_means.items()}
        if phase_total
        else {},
    )


def _metadata(backend, *, stage: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "backend": backend.name,
        "device": backend.device,
        "device_name": platform.processor() or platform.machine(),
        "numpy_version": np.__version__,
        "stage": stage,
        "method": (
            "one workload-cold synchronized update; warmup; consecutive "
            "per-update CUDA events resolved with one synchronization; "
            "independent device-synchronized steady throughput windows; "
            "epoch/full-run estimate = cold + steady * (total_updates - 1); "
            "repeat standard deviations extrapolate between-window steady-rate "
            "variation linearly and treat the single cold observation as fixed; "
            "optional separately synchronized phase pass; implemented "
            "post-update loss included"
        ),
    }
    if backend.is_gpu:
        cp = backend.xp
        device_index = int(backend.device.split(":", 1)[1])
        properties = cp.cuda.runtime.getDeviceProperties(device_index)
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        metadata.update(
            {
                "device_name": device_name,
                "cupy_version": cp.__version__,
                "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            }
        )
    else:
        metadata["cupy_version"] = None
        metadata["cuda_runtime_version"] = None
    return metadata


def _stage_value(
    explicit: int | None,
    *,
    stage: str,
    name: str,
) -> int:
    return int(STAGES[stage][name]) if explicit is None else explicit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        choices=CONDITIONS,
        dest="conditions",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Execution device: cpu or cuda:N (default: cuda:0).",
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAGES),
        default="estimate",
        help="update=quick precise estimate, estimate=stable estimate, detail=phases.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--measured-updates", type=int)
    parser.add_argument("--phase-updates", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    warmup_updates = _stage_value(
        args.warmup_updates, stage=args.stage, name="warmup_updates"
    )
    measured_updates = _stage_value(
        args.measured_updates, stage=args.stage, name="measured_updates"
    )
    phase_updates = _stage_value(
        args.phase_updates, stage=args.stage, name="phase_updates"
    )
    repetitions = _stage_value(args.repetitions, stage=args.stage, name="repetitions")
    if (
        min(
            args.batch_size,
            args.epochs,
            measured_updates,
            repetitions,
        )
        < 1
        or min(warmup_updates, phase_updates) < 0
    ):
        parser.error(
            "batch size, epochs, measured updates, and repetitions must be "
            "positive; warmup and phase updates must be non-negative"
        )

    backend, corpus, contexts, targets = _load_data(args.device)
    results = []
    for condition in args.conditions or CONDITIONS:
        backend.seed(1)
        np.random.seed(1)
        result = profile_condition(
            condition,
            corpus=corpus,
            contexts=contexts,
            targets=targets,
            backend=backend,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_updates=warmup_updates,
            measured_updates=measured_updates,
            phase_updates=phase_updates,
            repetitions=repetitions,
        )
        results.append(result)
        print(
            f"{condition}: {result.mean_ms_per_update:.3f} ± "
            f"{result.stdev_ms_per_update:.3f} ms/update, "
            f"{result.estimated_seconds_per_epoch:.1f} ± "
            f"{result.estimated_repeat_stdev_seconds_per_epoch:.1f} s/epoch, "
            f"{result.estimated_seconds_total:.1f} ± "
            f"{result.estimated_repeat_stdev_seconds_total:.1f} "
            f"s/{args.epochs} epochs",
            flush=True,
        )
    payload = {
        "schema_version": 6,
        "metadata": _metadata(backend, stage=args.stage),
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
