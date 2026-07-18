from __future__ import annotations

import matplotlib.pyplot as plt

from common import ANALYSIS_ROOT, client, latest_run_ids_for_atomic_ids, load_records, metric_history, parser, print_outputs, save_summary_csv


ATOMIC_RUN_IDS = ["MLP-SGD-HE", "MLP-MOM-HE", "MLP-ADAGRAD-HE", "MLP-ADAM-HE"]
OUTPUT = ANALYSIS_ROOT / "e02_mnist_mlp_optimizer.png"


def main() -> None:
    args = parser("Render e02 MNIST MLP optimizer results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    run_ids = args.run_id or latest_run_ids_for_atomic_ids(
        mlflow_client,
        experiment_name=args.mlflow_experiment,
        atomic_run_ids=ATOMIC_RUN_IDS,
        param_filters={"training/entrypoint": "experiments/run/e02_mnist_mlp_optimizer.py"},
    )
    records = load_records(mlflow_client, run_ids)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8))
    for record in records:
        for axis, metric in [(axes[0], "epoch/train/loss"), (axes[0], "epoch/test/loss")]:
            steps, values = metric_history(mlflow_client, run_id=record.mlflow_run_id, key=metric)
            if values:
                linestyle = "-" if metric.endswith("train/loss") else "--"
                axis.plot(steps, values, linestyle=linestyle, marker="o", markersize=3, label=f"{record.atomic_run_id} {metric.split('/')[1]}")
        for axis, metric in [(axes[1], "epoch/train/accuracy"), (axes[1], "epoch/test/accuracy")]:
            steps, values = metric_history(mlflow_client, run_id=record.mlflow_run_id, key=metric)
            if values:
                linestyle = "-" if metric.endswith("train/accuracy") else "--"
                axis.plot(steps, values, linestyle=linestyle, marker="o", markersize=3, label=f"{record.atomic_run_id} {metric.split('/')[1]}")

    axes[0].set_title("e02 loss by epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("e02 accuracy by epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.legend(fontsize=7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    save_summary_csv(
        args.summary_csv,
        records=records,
        param_keys=["optimizer/name", "optimizer/learning_rate"],
        metric_keys=[
            "final/train/loss",
            "final/test/loss",
            "final/train/accuracy",
            "final/test/accuracy",
            "runtime/train_total_s",
            "runtime/run_wall_total_s",
        ],
    )
    print_outputs(args.output, args.summary_csv, records)


if __name__ == "__main__":
    main()
