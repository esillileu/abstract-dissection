import matplotlib.pyplot as plt
import numpy as np

from .common import ANALYSIS_ROOT, client, latest_seeded_runs, parser


EXPERIMENT_ID = "e01"


def main() -> None:
    argument_parser = parser("Render the book-style e01 CBOW loss curve.", ANALYSIS_ROOT / "e01_cbow_book_reproduction.png")
    argument_parser.add_argument("--seed", type=int, default=1208965604, help="Completed seed trial to render as the book-style single curve.")
    args = argument_parser.parse_args()
    mlflow_client = client(args.tracking_uri)
    runs = latest_seeded_runs(mlflow_client, args.mlflow_experiment, ["W2V-CBOW-NS"])["W2V-CBOW-NS"]
    run = next((value for value in runs if int(value.data.params.get("seed/master", value.data.params.get("seed", -1))) == args.seed), None)
    if run is None:
        raise ValueError(f"missing completed W2V-CBOW-NS run for seed {args.seed}")
    history = sorted(
        mlflow_client.get_metric_history(run.info.run_id, "step/train/book_loss"),
        key=lambda value: value.step,
    )
    if not history:
        raise ValueError(f"run {run.info.run_id} has no step/train/book_loss history")
    figure, axis = plt.subplots()
    axis.plot(np.arange(len(history)), [value.value for value in history], label="train")
    axis.set_xlabel("iterations (x20)")
    axis.set_ylabel("loss")
    axis.set_title("e01 CBOW negative-sampling loss")
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
