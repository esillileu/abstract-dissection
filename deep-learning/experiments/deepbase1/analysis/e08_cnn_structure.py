from __future__ import annotations

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, client, latest_run_ids_for_atomic_ids, load_records, metric_history, parser, print_outputs, save_summary_csv


ATOMIC_RUN_IDS = ["CNN-SIMPLE", "CNN-DEEP"]
OUTPUT = ANALYSIS_ROOT / "e08_cnn_structure.png"


def main() -> None:
    args = parser("Render e08 CNN structure results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    run_ids = args.run_id or latest_run_ids_for_atomic_ids(
        mlflow_client,
        experiment_name=args.mlflow_experiment,
        atomic_run_ids=ATOMIC_RUN_IDS,
    )
    records = load_records(mlflow_client, run_ids)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8))
    for record in records:
        steps, values = metric_history(mlflow_client, run_id=record.mlflow_run_id, key="epoch/test/accuracy")
        if values:
            axes[0].plot(steps, values, marker="o", label=record.atomic_run_id)

    labels = [record.atomic_run_id for record in records]
    accuracy = [record.metrics.get("final/test/accuracy", 0.0) for record in records]
    runtime = [record.metrics.get("runtime/train_total_s", 0.0) for record in records]
    x = range(len(records))
    axes[1].bar([value - 0.18 for value in x], accuracy, width=0.36, label="test accuracy")
    runtime_axis = axes[1].twinx()
    runtime_axis.bar([value + 0.18 for value in x], runtime, width=0.36, color="tab:orange", label="train seconds")
    axes[1].set_xticks(list(x), labels)

    axes[0].set_title("e08 test accuracy by epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("e08 final accuracy and runtime")
    axes[1].set_ylabel("accuracy")
    runtime_axis.set_ylabel("seconds")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left", fontsize=8)
    runtime_axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    save_summary_csv(
        args.summary_csv,
        records=records,
        param_keys=["model/name", "model/num_conv_layers"],
        metric_keys=[
            "final/train/loss",
            "final/test/loss",
            "final/train/accuracy",
            "final/test/accuracy",
            "model/flops",
            "model/macs",
            "runtime/train_total_s",
            "runtime/run_wall_total_s",
        ],
    )
    print_outputs(args.output, args.summary_csv, records)


if __name__ == "__main__":
    main()
