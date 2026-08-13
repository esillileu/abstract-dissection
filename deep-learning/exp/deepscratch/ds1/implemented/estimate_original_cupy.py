"""Estimate full runtime after replacing the book code's NumPy with CuPy."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from .train_original import BOOK_ROOT, _add_book_to_import_path


DEFAULT_OUTPUT = Path("exp/deepscratch/ds1/original/legacy_results/fixed_seed/cupy_estimate.json")
UPDATES_PER_EPOCH = 600
EPOCHS = 20
TOTAL_UPDATES = UPDATES_PER_EPOCH * EPOCHS
NUMPY_MODULES = (
    "common.functions",
    "common.gradient",
    "common.layers",
    "common.optimizer",
    "common.trainer",
    "common.util",
    "ch07.simple_convnet",
    "ch08.deep_convnet",
)


@dataclass(frozen=True)
class RuntimeEstimate:
    experiment_id: str
    model: str
    benchmark_updates: int
    seconds_per_update: float
    projected_update_time_s: float
    epoch_probe_time_s: float
    projected_epoch_probe_time_s: float
    final_test_time_s: float
    projected_total_time_s: float


def estimate(experiment_id: str, *, benchmark_updates: int) -> RuntimeEstimate:
    cp = importlib.import_module("cupy")
    _add_book_to_import_path()

    from dataset.mnist import load_mnist

    (x_train_np, t_train_np), (x_test_np, t_test_np) = load_mnist(flatten=False)
    modules = {
        name: importlib.import_module(name)
        for name in NUMPY_MODULES
    }
    for module in modules.values():
        if hasattr(module, "np"):
            module.np = cp

    x_train = cp.asarray(x_train_np)
    t_train = cp.asarray(t_train_np)
    x_test = cp.asarray(x_test_np)
    t_test = cp.asarray(t_test_np)
    cp.random.seed(1)

    if experiment_id == "e06":
        network = modules["ch07.simple_convnet"].SimpleConvNet(
            input_dim=(1, 28, 28),
            conv_param={
                "filter_num": 30,
                "filter_size": 5,
                "pad": 0,
                "stride": 1,
            },
            hidden_size=100,
            output_size=10,
            weight_init_std=0.01,
        )
    elif experiment_id == "e07":
        network = modules["ch08.deep_convnet"].DeepConvNet()
    else:
        raise ValueError(f"unknown experiment: {experiment_id}")

    trainer = modules["common.trainer"].Trainer(
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=EPOCHS,
        mini_batch_size=100,
        optimizer="Adam",
        optimizer_param={"lr": 0.001},
        evaluate_sample_num_per_epoch=1000,
        verbose=False,
    )

    # Avoid Trainer's epoch probe at current_iter == 0 while timing updates.
    trainer.current_iter = 1
    trainer.train_step()
    cp.cuda.get_current_stream().synchronize()

    started = perf_counter()
    for _ in range(benchmark_updates):
        trainer.train_step()
    cp.cuda.get_current_stream().synchronize()
    seconds_per_update = (perf_counter() - started) / benchmark_updates

    # Warm up and time the two 1,000-example probes performed every epoch.
    network.accuracy(x_train[:1000], t_train[:1000])
    network.accuracy(x_test[:1000], t_test[:1000])
    cp.cuda.get_current_stream().synchronize()
    started = perf_counter()
    network.accuracy(x_train[:1000], t_train[:1000])
    network.accuracy(x_test[:1000], t_test[:1000])
    cp.cuda.get_current_stream().synchronize()
    epoch_probe_time_s = perf_counter() - started

    network.accuracy(x_test, t_test)
    cp.cuda.get_current_stream().synchronize()
    started = perf_counter()
    network.accuracy(x_test, t_test)
    cp.cuda.get_current_stream().synchronize()
    final_test_time_s = perf_counter() - started

    update_time = seconds_per_update * TOTAL_UPDATES
    all_epoch_probes = epoch_probe_time_s * EPOCHS
    return RuntimeEstimate(
        experiment_id=experiment_id,
        model=type(network).__name__,
        benchmark_updates=benchmark_updates,
        seconds_per_update=seconds_per_update,
        projected_update_time_s=update_time,
        epoch_probe_time_s=epoch_probe_time_s,
        projected_epoch_probe_time_s=all_epoch_probes,
        final_test_time_s=final_test_time_s,
        projected_total_time_s=update_time + all_epoch_probes + final_test_time_s,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.updates < 1:
        parser.error("--updates must be positive")

    results = [
        estimate(experiment_id, benchmark_updates=args.updates)
        for experiment_id in ("e06", "e07")
    ]
    payload = {
        "method": (
            "Original book modules with their module-level np names replaced "
            "by cupy; 12,000 updates plus original evaluation schedule."
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
            f"{result.experiment_id}: "
            f"{result.seconds_per_update:.6f}s/update, "
            f"projected={result.projected_total_time_s:.1f}s",
            flush=True,
        )
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
