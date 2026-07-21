from __future__ import annotations

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, client, latest_seeded_records, metric_curve, parser, plot_curve, print_outputs, save_summary_csv


EXPERIMENT_ID = "e02"
ATOMIC_RUN_IDS = ["MLP-SGD-HE", "MLP-MOM-HE", "MLP-ADAGRAD-HE", "MLP-ADAM-HE"]
OUTPUT = ANALYSIS_ROOT / "e02_mnist_mlp_optimizer.png"


def main() -> None:
    args = parser("Render e02 MNIST MLP optimizer results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    grouped = latest_seeded_records(mlflow_client, experiment_name=args.mlflow_experiment, atomic_run_ids=ATOMIC_RUN_IDS)
    records = [record for values in grouped.values() for record in values]

    fig, axis = plt.subplots(figsize=(9, 5))
    for atomic_run_id in ATOMIC_RUN_IDS:
        plot_curve(axis, metric_curve(mlflow_client, grouped[atomic_run_id], "update/train/loss"), label=atomic_run_id, marker="o")

    axis.set_title("e02 optimizer comparison")
    axis.set_xlabel("update")
    axis.set_ylabel("loss")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
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
