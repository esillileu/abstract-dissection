"""Run the unmodified book CNN implementations and save timing/accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


BOOK_ROOT = Path(
    "01_deep-learning-from-base/WegraLee-deep-learning-from-scratch"
).resolve()
DEFAULT_OUTPUT = Path("exp/ds1/results/original")
EXPERIMENTS = ("e06", "e07")


@dataclass(frozen=True)
class OriginalResult:
    experiment_id: str
    source: str
    model: str
    backend: str
    seed: int
    epochs: int
    updates: int
    batch_size: int
    training_time_s: float
    train_accuracy: float
    test_accuracy: float


def train_original(
    experiment_id: str,
    *,
    seed: int = 1,
    epochs: int = 20,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> OriginalResult:
    """Run the book's model and Trainer classes with their original settings."""
    _add_book_to_import_path()
    from common.trainer import Trainer
    from dataset.mnist import load_mnist

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    if train_limit is not None:
        x_train, t_train = x_train[:train_limit], t_train[:train_limit]
    if test_limit is not None:
        x_test, t_test = x_test[:test_limit], t_test[:test_limit]

    np.random.seed(seed)
    if experiment_id == "e06":
        from ch07.simple_convnet import SimpleConvNet

        network = SimpleConvNet(
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
        source = "ch07/train_convnet.py"
    elif experiment_id == "e07":
        from ch08.deep_convnet import DeepConvNet

        network = DeepConvNet()
        source = "ch08/train_deepnet.py"
    else:
        raise ValueError(f"unknown experiment: {experiment_id}")

    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=epochs,
        mini_batch_size=100,
        optimizer="Adam",
        optimizer_param={"lr": 0.001},
        evaluate_sample_num_per_epoch=1000,
        verbose=False,
    )
    started = perf_counter()
    trainer.train()
    training_time_s = perf_counter() - started

    return OriginalResult(
        experiment_id=experiment_id,
        source=source,
        model=type(network).__name__,
        backend="numpy-cpu",
        seed=seed,
        epochs=epochs,
        updates=trainer.max_iter,
        batch_size=trainer.batch_size,
        training_time_s=training_time_s,
        train_accuracy=float(network.accuracy(x_train, t_train)),
        test_accuracy=float(network.accuracy(x_test, t_test)),
    )


def save_results(results: list[OriginalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    for result, row in zip(results, rows, strict=True):
        (output_dir / f"{result.experiment_id}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with (output_dir / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--experiment",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    results = []
    for experiment_id in args.experiment:
        print(f"{experiment_id}: original NumPy training started", flush=True)
        result = train_original(experiment_id, seed=args.seed)
        results.append(result)
        save_results(results, args.output_dir)
        print(
            f"{experiment_id}: {result.training_time_s:.3f}s, "
            f"test_accuracy={result.test_accuracy:.4%}",
            flush=True,
        )
    print(f"saved: {args.output_dir}", flush=True)


def _add_book_to_import_path() -> None:
    book_root = str(BOOK_ROOT)
    if book_root not in sys.path:
        sys.path.insert(0, book_root)


if __name__ == "__main__":
    main()
