"""Extension figure for the book's commented Skip-gram alternative."""

from .common import ANALYSIS_ROOT, client, parser
from .render import render


EXPERIMENT_ID = "e01"


def main() -> None:
    args = parser("Render the e01 CBOW/Skip-gram extension comparison.", ANALYSIS_ROOT / "e01_cbow_skipgram_comparison.png").parse_args()
    render(mlflow_client=client(args.tracking_uri), experiment_name=args.mlflow_experiment, atomic_run_ids=["W2V-CBOW-NS", "W2V-SG-NS"], metric="step/train/normalized_loss", title="e01 CBOW vs Skip-gram (extension)", ylabel="normalized loss", output=args.output, summary_csv=args.summary_csv)


if __name__ == "__main__":
    main()
