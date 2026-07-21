from __future__ import annotations

import csv

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, ErrorBarStyle, client, latest_seeded_records, metric_curve, parser, plot_curve, print_outputs, save_summary_csv


EXPERIMENT_ID = "e08"
MODEL_PAIRS = {
    "ParameterMatchedNN": ("NN-MATCHED", "NN-MATCHED-PERMUTED"),
    "SimpleConvNet": ("CNN-SIMPLE", "CNN-SIMPLE-PERMUTED"),
}
ATOMIC_RUN_IDS = [atomic_run_id for pair in MODEL_PAIRS.values() for atomic_run_id in pair]
OUTPUT = ANALYSIS_ROOT / "e08_spatial_layout.png"
ERROR_BARS = ErrorBarStyle(every=2)


def _seed(record) -> str:
    return record.params.get("seed/master", record.params.get("seed", record.mlflow_run_id))


def _write_paired_final_drops(path, grouped) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "original_run_id", "permuted_run_id", "original_test_accuracy", "permuted_test_accuracy", "permutation_drop"])
        writer.writeheader()
        for model, (original_id, permuted_id) in MODEL_PAIRS.items():
            original = {_seed(record): record for record in grouped[original_id]}
            permuted = {_seed(record): record for record in grouped[permuted_id]}
            if original.keys() != permuted.keys():
                raise ValueError(f"{model} original/permuted seed sets differ")
            for seed in sorted(original):
                original_accuracy = original[seed].metrics["final/test/accuracy"]
                permuted_accuracy = permuted[seed].metrics["final/test/accuracy"]
                writer.writerow({"model": model, "seed": seed, "original_run_id": original[seed].mlflow_run_id, "permuted_run_id": permuted[seed].mlflow_run_id, "original_test_accuracy": original_accuracy, "permuted_test_accuracy": permuted_accuracy, "permutation_drop": original_accuracy - permuted_accuracy})


def main() -> None:
    args = parser("Render e08 spatial-layout learning curves.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    grouped = latest_seeded_records(mlflow_client, experiment_name=args.mlflow_experiment, atomic_run_ids=ATOMIC_RUN_IDS)
    records = [record for values in grouped.values() for record in values]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, (model, (original_id, permuted_id)) in zip(axes, MODEL_PAIRS.items(), strict=True):
        plot_curve(axis, metric_curve(mlflow_client, grouped[original_id], "epoch/test/accuracy"), label="original", marker="o", error_bars=ERROR_BARS)
        plot_curve(axis, metric_curve(mlflow_client, grouped[permuted_id], "epoch/test/accuracy"), label="pixel-permuted", linestyle="--", marker="s", error_bars=ERROR_BARS)
        axis.set_title(model)
        axis.set_xlabel("epoch")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("full-test accuracy")
    fig.suptitle("e08 spatial-layout dependence")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    save_summary_csv(args.summary_csv, records=records, param_keys=["model/name", "dataset/input_transform/name"], metric_keys=["final/test/accuracy", "final/train/accuracy", "final/test/loss", "final/status/success"])
    _write_paired_final_drops(args.output.with_name("e08_spatial_layout_paired_drops.csv"), grouped)
    print_outputs(args.output, args.summary_csv, records)


if __name__ == "__main__":
    main()
