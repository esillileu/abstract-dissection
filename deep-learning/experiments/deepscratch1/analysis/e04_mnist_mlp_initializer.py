from __future__ import annotations

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, ErrorBarStyle, client, latest_seeded_records, parser, plot_curve, print_outputs, save_summary_csv, smoothed_step_loss_curve


EXPERIMENT_ID = "e04"
ATOMIC_RUN_IDS = ["MLP-SGD-STD001", "MLP-SGD-XAVIER", "MLP-SGD-HE"]
OUTPUT = ANALYSIS_ROOT / "e04_mnist_mlp_initializer.png"
ERROR_BARS = ErrorBarStyle(every=20)
MARKERS = {"MLP-SGD-STD001": "o", "MLP-SGD-XAVIER": "s", "MLP-SGD-HE": "D"}


def main() -> None:
    args = parser("Render e04 MNIST MLP initializer results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    grouped = latest_seeded_records(mlflow_client, experiment_name=args.mlflow_experiment, atomic_run_ids=ATOMIC_RUN_IDS)
    records = [record for values in grouped.values() for record in values]

    fig, axis = plt.subplots(figsize=(9, 5))
    for atomic_run_id in ATOMIC_RUN_IDS:
        plot_curve(axis, smoothed_step_loss_curve(mlflow_client, grouped[atomic_run_id]), label=atomic_run_id, marker=MARKERS[atomic_run_id], error_bars=ERROR_BARS)

    axis.set_title("e04 weight initialization comparison")
    axis.set_xlabel("update")
    axis.set_ylabel("loss")
    axis.set_ylim(0, 2.5)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    save_summary_csv(
        args.summary_csv,
        records=records,
        param_keys=["initializer/name"],
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
