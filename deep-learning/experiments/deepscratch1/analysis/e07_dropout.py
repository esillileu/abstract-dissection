from __future__ import annotations

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, ErrorBarStyle, client, latest_seeded_records, metric_curve, parser, plot_curve, print_outputs, save_summary_csv


EXPERIMENT_ID = "e07"
ATOMIC_RUN_IDS = ["REG-DO-02"]
OUTPUT = ANALYSIS_ROOT / "e07_dropout.png"
ERROR_BARS = ErrorBarStyle(every=5)


def dropout_label(record) -> str:
    ratio = record.params.get("regularization/dropout_ratio", "0.0")
    return f"{record.atomic_run_id} ({ratio})"


def main() -> None:
    args = parser("Render e07 dropout results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    grouped = latest_seeded_records(mlflow_client, experiment_name=args.mlflow_experiment, atomic_run_ids=ATOMIC_RUN_IDS)
    records = [record for values in grouped.values() for record in values]

    fig, axis = plt.subplots(figsize=(9, 5))
    for atomic_run_id in ATOMIC_RUN_IDS:
        label = dropout_label(grouped[atomic_run_id][0])
        plot_curve(axis, metric_curve(mlflow_client, grouped[atomic_run_id], "book_epoch/train/accuracy"), label=f"{label} train", marker="o", error_bars=ERROR_BARS)
        plot_curve(axis, metric_curve(mlflow_client, grouped[atomic_run_id], "book_epoch/test/accuracy"), label=f"{label} test", linestyle="--", marker="s", error_bars=ERROR_BARS)

    axis.set_title("e07 dropout")
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
