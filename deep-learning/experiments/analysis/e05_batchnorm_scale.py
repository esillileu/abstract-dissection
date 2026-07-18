from __future__ import annotations

import matplotlib.pyplot as plt

from common import ANALYSIS_ROOT, client, latest_run_ids_for_atomic_ids, load_records, parser, print_outputs, save_summary_csv


ATOMIC_RUN_IDS = [f"BN-OFF-{index:02d}" for index in range(1, 17)] + [f"BN-ON-{index:02d}" for index in range(1, 17)]
OUTPUT = ANALYSIS_ROOT / "e05_batchnorm_scale.png"


def scale_value(record) -> float:
    return float(record.params["initializer/scale"])


def main() -> None:
    args = parser("Render e05 BatchNorm scale results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    run_ids = args.run_id or latest_run_ids_for_atomic_ids(
        mlflow_client,
        experiment_name=args.mlflow_experiment,
        atomic_run_ids=ATOMIC_RUN_IDS,
    )
    records = load_records(mlflow_client, run_ids)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8))
    for prefix, label in [("BN-OFF", "BatchNorm off"), ("BN-ON", "BatchNorm on")]:
        group = sorted((record for record in records if record.atomic_run_id.startswith(prefix)), key=scale_value)
        scales = [scale_value(record) for record in group]
        accuracies = [record.metrics.get("final/test/accuracy", 0.0) for record in group]
        losses = [record.metrics.get("final/test/loss", 0.0) for record in group]
        axes[0].plot(scales, accuracies, marker="o", label=label)
        axes[1].plot(scales, losses, marker="o", label=label)

    axes[0].set_title("e05 final test accuracy by init scale")
    axes[0].set_xlabel("initializer scale")
    axes[0].set_ylabel("test accuracy")
    axes[1].set_title("e05 final test loss by init scale")
    axes[1].set_xlabel("initializer scale")
    axes[1].set_ylabel("test loss")
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
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
