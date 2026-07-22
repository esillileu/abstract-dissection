from __future__ import annotations

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, ErrorBarStyle, client, latest_seeded_records, metric_curve, parser, plot_curve, print_outputs, save_summary_csv


EXPERIMENT_ID = "e05"
ATOMIC_RUN_IDS = [f"BN-OFF-{index:02d}" for index in range(1, 17)] + [f"BN-ON-{index:02d}" for index in range(1, 17)]
OUTPUT = ANALYSIS_ROOT / "e05_batchnorm_scale.png"
ERROR_BARS = ErrorBarStyle(every=1)


def scale_value(record) -> float:
    return float(record.params["initializer/scale"])


def main() -> None:
    args = parser("Render e05 BatchNorm scale results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    grouped = latest_seeded_records(mlflow_client, experiment_name=args.mlflow_experiment, atomic_run_ids=ATOMIC_RUN_IDS)
    records = [record for values in grouped.values() for record in values]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(14, 11), sharex=True, sharey=True)
    for index, axis in enumerate(axes.flat, start=1):
        off_id, on_id = f"BN-OFF-{index:02d}", f"BN-ON-{index:02d}"
        scale = scale_value(grouped[on_id][0])
        plot_curve(axis, metric_curve(mlflow_client, grouped[on_id], "book_epoch/train/accuracy"), label="Batch Normalization", marker="o", error_bars=ERROR_BARS)
        plot_curve(axis, metric_curve(mlflow_client, grouped[off_id], "book_epoch/train/accuracy"), label="Normal (without BatchNorm)", linestyle="--", marker="s", error_bars=ERROR_BARS)
        axis.set_title(f"W: {scale:g}")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        if index % 4 == 1:
            axis.set_ylabel("accuracy")
        if index > 12:
            axis.set_xlabel("epoch")
    axes.flat[-1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    save_summary_csv(
        args.summary_csv,
        records=records,
        param_keys=["initializer/scale", "model/use_batchnorm"],
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
