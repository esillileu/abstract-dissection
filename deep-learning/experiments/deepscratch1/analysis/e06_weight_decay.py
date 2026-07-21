from __future__ import annotations

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, client, latest_seeded_records, metric_curve, parser, plot_curve, print_outputs, save_summary_csv


EXPERIMENT_ID = "e06"
ATOMIC_RUN_IDS = ["REG-WD-1E1"]
OUTPUT = ANALYSIS_ROOT / "e06_weight_decay.png"


def main() -> None:
    args = parser("Render e06 weight decay results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    grouped = latest_seeded_records(mlflow_client, experiment_name=args.mlflow_experiment, atomic_run_ids=ATOMIC_RUN_IDS)
    records = [record for values in grouped.values() for record in values]

    fig, axis = plt.subplots(figsize=(9, 5))
    for atomic_run_id in ATOMIC_RUN_IDS:
        plot_curve(axis, metric_curve(mlflow_client, grouped[atomic_run_id], "book_epoch/train/accuracy"), label=f"{atomic_run_id} train", marker="o")
        plot_curve(axis, metric_curve(mlflow_client, grouped[atomic_run_id], "book_epoch/test/accuracy"), label=f"{atomic_run_id} test", linestyle="--", marker="s")

    axis.set_title("e06 weight decay")
    axis.set_xlabel("epoch")
    axis.set_ylabel("accuracy")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    save_summary_csv(
        args.summary_csv,
        records=records,
        param_keys=["optimizer/weight_decay"],
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
