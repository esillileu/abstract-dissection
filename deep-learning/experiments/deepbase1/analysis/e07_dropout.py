from __future__ import annotations

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, client, latest_run_ids_for_atomic_ids, load_records, metric_history, parser, print_outputs, save_summary_csv


ATOMIC_RUN_IDS = ["REG-BASE", "REG-DO-01", "REG-DO-02", "REG-DO-03", "REG-DO-05"]
OUTPUT = ANALYSIS_ROOT / "e07_dropout.png"


def dropout_label(record) -> str:
    ratio = record.params.get("regularization/dropout_ratio", "0.0")
    return f"{record.atomic_run_id} ({ratio})"


def main() -> None:
    args = parser("Render e07 dropout results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    run_ids = args.run_id or latest_run_ids_for_atomic_ids(
        mlflow_client,
        experiment_name=args.mlflow_experiment,
        atomic_run_ids=ATOMIC_RUN_IDS,
    )
    records = load_records(mlflow_client, run_ids)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8))
    for record in records:
        steps, train = metric_history(mlflow_client, run_id=record.mlflow_run_id, key="epoch/train/accuracy")
        _, test = metric_history(mlflow_client, run_id=record.mlflow_run_id, key="epoch/test/accuracy")
        if train:
            axes[0].plot(steps, train, label=f"{record.atomic_run_id} train", alpha=0.75)
        if test:
            axes[0].plot(steps, test, linestyle="--", label=f"{record.atomic_run_id} test", alpha=0.75)

    labels = [dropout_label(record) for record in records]
    gap = [
        record.metrics.get("final/train/accuracy", 0.0) - record.metrics.get("final/test/accuracy", 0.0)
        for record in records
    ]
    axes[1].bar(range(len(records)), gap)
    axes[1].set_xticks(list(range(len(records))), labels, rotation=25, ha="right")

    axes[0].set_title("e07 accuracy by epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("e07 train-test accuracy gap")
    axes[1].set_ylabel("gap")
    axes[1].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    save_summary_csv(
        args.summary_csv,
        records=records,
        param_keys=["regularization/dropout_ratio"],
        metric_keys=[
            "final/train/loss",
            "final/test/loss",
            "final/train/accuracy",
            "final/test/accuracy",
            "runtime/train_total_s",
        ],
    )
    print_outputs(args.output, args.summary_csv, records)


if __name__ == "__main__":
    main()
