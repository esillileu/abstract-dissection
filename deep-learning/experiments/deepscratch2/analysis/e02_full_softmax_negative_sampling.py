from .common import ANALYSIS_ROOT, client, parser
from .render import render


EXPERIMENT_ID = "e02"


def main() -> None:
    args = parser("Render e02 architecture/objective results.", ANALYSIS_ROOT / "e02_full_softmax_negative_sampling.png").parse_args()
    render(mlflow_client=client(args.tracking_uri), experiment_name=args.mlflow_experiment, atomic_run_ids=["W2V-CBOW-FULL", "W2V-CBOW-NS", "W2V-SG-FULL", "W2V-SG-NS"], metric="step/train/normalized_loss", title="e02 Full softmax vs negative sampling by architecture", ylabel="normalized loss", output=args.output, summary_csv=args.summary_csv)


if __name__ == "__main__":
    main()
