from .common import ANALYSIS_ROOT, client, parser
from .render import render


def main() -> None:
    args = parser("Render e09 CBOW and Skip-gram results.", ANALYSIS_ROOT / "e09_cbow_skipgram.png").parse_args()
    render(mlflow_client=client(args.tracking_uri), experiment_name=args.mlflow_experiment, atomic_run_ids=["W2V-CBOW-NS", "W2V-SG-NS"], metric="step/train/normalized_loss", title="e09 normalized loss", ylabel="normalized loss", output=args.output, summary_csv=args.summary_csv)


if __name__ == "__main__":
    main()
