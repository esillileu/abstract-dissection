from __future__ import annotations

import matplotlib.pyplot as plt

from .common import ANALYSIS_ROOT, client, latest_run_ids_for_atomic_ids, load_records, parser, print_outputs, save_summary_csv


ACTIVATIONS = ["SIG", "TANH", "RELU"]
INITIALIZERS = ["STD1", "STD001", "XAVIER", "HE"]
ATOMIC_RUN_IDS = [f"ACT-{activation}-{initializer}" for activation in ACTIVATIONS for initializer in INITIALIZERS]
OUTPUT = ANALYSIS_ROOT / "e03_activation_probe.png"


def main() -> None:
    args = parser("Render e03 activation probe results.", OUTPUT).parse_args()
    mlflow_client = client(args.tracking_uri)
    run_ids = args.run_id or latest_run_ids_for_atomic_ids(
        mlflow_client,
        experiment_name=args.mlflow_experiment,
        atomic_run_ids=ATOMIC_RUN_IDS,
    )
    records = load_records(mlflow_client, run_ids)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8))
    for record in records:
        layer_steps = list(range(1, 6))
        stds = [record.metrics.get(f"layer/{index:02d}/activation/std") for index in layer_steps]
        means = [record.metrics.get(f"layer/{index:02d}/activation/mean") for index in layer_steps]
        if all(value is not None for value in stds):
            axes[0].plot(layer_steps, stds, marker="o", linewidth=1.6, label=record.atomic_run_id)
        if all(value is not None for value in means):
            axes[1].plot(layer_steps, means, marker="o", linewidth=1.6, label=record.atomic_run_id)

    axes[0].set_title("e03 activation std by layer")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("std")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("e03 activation mean by layer")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("mean")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        axis.legend(fontsize=6, ncols=2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    save_summary_csv(
        args.summary_csv,
        records=records,
        param_keys=["model/activation", "initializer/name", "initializer/scale"],
        metric_keys=[
            "final/activation/std_retention_ratio",
            "final/activation/mean_absolute_shift",
            "final/activation/max_saturation_ratio",
            "final/activation/max_zero_ratio",
            "runtime/train_total_s",
        ],
    )
    print_outputs(args.output, args.summary_csv, records)


if __name__ == "__main__":
    main()
