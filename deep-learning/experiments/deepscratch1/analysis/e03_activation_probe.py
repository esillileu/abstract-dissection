from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mlprosection.experiment import load_yaml
from mlprosection.experiment.executors.activation_probe import _activate, _std

from .common import ANALYSIS_ROOT, client, latest_seeded_records, parser, print_outputs, save_summary_csv


EXPERIMENT_ID = "e03"
CONFIG = Path("experiments/deepscratch1/config/e03_activation_probe.yaml")
ATOMIC_RUN_IDS = ["ACT-SIG-STD1"]
OUTPUT = ANALYSIS_ROOT / "e03_activation_probe.png"


def _seed(record) -> int:
    value = record.params.get("seed/master", record.params.get("seed"))
    if value is None:
        raise ValueError(f"e03 run is missing its master seed: {record.mlflow_run_id}")
    return int(value)


def activation_layers(atomic_run_id: str, seed: int) -> tuple[str, list[np.ndarray]]:
    config = load_yaml(CONFIG, atomic_run_id=atomic_run_id)
    model, initializer = config["model"], config["initializer"]
    assert isinstance(model, dict) and isinstance(initializer, dict)
    width, depth = int(model["width"]), int(model["depth"])
    samples = int(config["dataset"]["train_size"])
    activation = str(model["activation"])
    dtype = np.dtype(str(config["numerics"]["dtype"]))
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((samples, width), dtype=dtype)
    layers = []
    for _ in range(depth):
        weights = rng.standard_normal((width, width), dtype=dtype) * _std(str(initializer["name"]), width, initializer)
        values = _activate(activation, values @ weights)
        layers.append(values)
    return activation, layers


def render_histograms(grouped, output: Path) -> None:
    figure, axes = plt.subplots(nrows=len(ATOMIC_RUN_IDS), ncols=5, figsize=(15, 24), squeeze=False)
    for row, atomic_run_id in enumerate(ATOMIC_RUN_IDS):
        seed_layers = [activation_layers(atomic_run_id, _seed(record)) for record in grouped[atomic_run_id]]
        activation = seed_layers[0][0]
        for column, axis in enumerate(axes[row]):
            samples = [layers[column].ravel() for _, layers in seed_layers]
            merged = np.concatenate(samples)
            value_range = (0.0, 1.0) if activation == "sigmoid" else (float(merged.min()), float(merged.max()))
            counts = np.asarray([np.histogram(values, bins=30, range=value_range)[0] for values in samples])
            edges = np.linspace(*value_range, 31)
            centers, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
            mean, minimum, maximum = counts.mean(axis=0), counts.min(axis=0), counts.max(axis=0)
            axis.bar(centers, mean, width=widths, alpha=0.55, color="tab:blue")
            axis.errorbar(centers, mean, yerr=np.vstack((mean - minimum, maximum - mean)), fmt="none", ecolor="tab:blue", capsize=1.5, linewidth=0.8)
            if row == 0:
                axis.set_title(f"{column + 1}-layer")
            if row == len(ATOMIC_RUN_IDS) - 1:
                axis.set_xlabel("activation")
            if column == 0:
                axis.set_ylabel(f"{atomic_run_id}\ncount", fontsize=8)
    figure.suptitle("e03 activation distributions: one condition per row", fontsize=15, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_condition(atomic_run_id: str, seed_layers: list[tuple[str, list[np.ndarray]]], output: Path) -> None:
    activation = seed_layers[0][0]
    figure, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3.4), sharey=True)
    for column, axis in enumerate(axes):
        samples = [layers[column].ravel() for _, layers in seed_layers]
        merged = np.concatenate(samples)
        value_range = (0.0, 1.0) if activation == "sigmoid" else (float(merged.min()), float(merged.max()))
        counts = np.asarray([np.histogram(values, bins=30, range=value_range)[0] for values in samples])
        edges = np.linspace(*value_range, 31)
        centers, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
        mean, minimum, maximum = counts.mean(axis=0), counts.min(axis=0), counts.max(axis=0)
        axis.bar(centers, mean, width=widths, alpha=0.55, color="tab:blue")
        axis.errorbar(centers, mean, yerr=np.vstack((mean - minimum, maximum - mean)), fmt="none", ecolor="tab:blue", capsize=1.5, linewidth=0.8)
        axis.set_title(f"{column + 1}-layer")
        axis.set_xlabel("activation")
        if column == 0:
            axis.set_ylabel("count")
    figure.suptitle(f"{atomic_run_id}: activation distribution (n={len(seed_layers)})")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def condition_output_path(output: Path, atomic_run_id: str) -> Path:
    return output.with_name(f"{output.stem}_{atomic_run_id.lower()}{output.suffix}")


def main() -> None:
    argument_parser = parser("Render e03 activation distributions.", OUTPUT)
    argument_parser.add_argument("--layout", choices=("combined", "individual"), default="combined")
    args = argument_parser.parse_args()
    mlflow_client = client(args.tracking_uri)
    grouped = latest_seeded_records(mlflow_client, experiment_name=args.mlflow_experiment, atomic_run_ids=ATOMIC_RUN_IDS)
    records = [record for values in grouped.values() for record in values]
    if args.layout == "combined":
        render_histograms(grouped, args.output)
        print(f"output={args.output}")
    else:
        for atomic_run_id in ATOMIC_RUN_IDS:
            seed_layers = [activation_layers(atomic_run_id, _seed(record)) for record in grouped[atomic_run_id]]
            output = condition_output_path(args.output, atomic_run_id)
            render_condition(atomic_run_id, seed_layers, output)
            print(f"output={output}")

    save_summary_csv(
        args.summary_csv,
        records=records,
        param_keys=["model/activation", "initializer/name", "initializer/scale"],
        metric_keys=[
            "final/activation/std_retention_ratio",
            "final/activation/mean_absolute_shift",
            "final/activation/max_saturation_ratio",
            "final/activation/max_zero_ratio",
        ],
    )
    print_outputs(args.output, args.summary_csv, records)


if __name__ == "__main__":
    main()
