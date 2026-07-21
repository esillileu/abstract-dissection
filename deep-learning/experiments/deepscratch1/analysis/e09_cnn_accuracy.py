from __future__ import annotations

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, ErrorBarStyle, client, latest_seeded_records, metric_curve, parser, plot_curve, print_outputs, save_summary_csv


EXPERIMENT_ID = "e09"
RUNS = {
    "CNN-SIMPLE": {"group": "g08", "target": 0.9896, "label": "SimpleConvNet"},
    "CNN-DEEP-ACCURACY": {"group": "g10", "target": 0.9938, "label": "DeepConvNet"},
}
OUTPUT = ANALYSIS_ROOT / "e09_cnn_accuracy.png"
ERROR_BARS = ErrorBarStyle(every=2)


def main() -> None:
    args = parser("Render e09 CNN full-test accuracy results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    grouped = {}
    for atomic_run_id, definition in RUNS.items():
        records_for_id = latest_seeded_records(
            mlflow_client,
            experiment_name=args.mlflow_experiment,
            atomic_run_ids=[atomic_run_id],
        )[atomic_run_id]
        grouped[atomic_run_id] = [
            record for record in records_for_id
            if record.tags.get("execution_group.id") == definition["group"]
        ]
        if not grouped[atomic_run_id]:
            raise ValueError(f"missing {atomic_run_id} runs for {definition['group']}")
    records = [record for values in grouped.values() for record in values]

    fig, axis = plt.subplots(figsize=(8, 5))
    for atomic_run_id, definition in RUNS.items():
        plot_curve(axis, metric_curve(mlflow_client, grouped[atomic_run_id], "epoch/test/accuracy"), label=definition["label"], marker="o" if atomic_run_id == "CNN-SIMPLE" else "s", error_bars=ERROR_BARS)
        axis.axhline(definition["target"], color="0.45", linestyle=":" if atomic_run_id == "CNN-SIMPLE" else "--", linewidth=1, label=f"{definition['label']} target ({definition['target']:.2%})")

    axis.set_title("e09 CNN accuracy reproduction")
    axis.set_xlabel("epoch")
    axis.set_ylabel("full-test accuracy")
    axis.set_ylim(0.9, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    save_summary_csv(args.summary_csv, records=records, param_keys=["model/name", "execution_group_id"], metric_keys=["final/test/accuracy", "final/train/accuracy", "final/status/success"])
    print_outputs(args.output, args.summary_csv, records)


if __name__ == "__main__":
    main()
