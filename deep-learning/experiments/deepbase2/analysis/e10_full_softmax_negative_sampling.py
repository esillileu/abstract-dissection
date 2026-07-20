from .common import ANALYSIS_ROOT, client, parser
from .render import render


def main() -> None:
    args = parser("Render e10 output-objective results.", ANALYSIS_ROOT / "e10_full_softmax_negative_sampling.png").parse_args()
    render(mlflow_client=client(args.tracking_uri), experiment_name=args.mlflow_experiment, atomic_run_ids=["W2V-CBOW-FULL", "W2V-CBOW-NS"], metric="step/train/normalized_loss", title="e10 normalized loss", ylabel="normalized loss", output=args.output, summary_csv=args.summary_csv)


if __name__ == "__main__":
    main()
