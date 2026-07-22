from .common import ANALYSIS_ROOT, client, parser
from .render import render


EXPERIMENT_ID = "e06"


def main() -> None:
    args = parser("Render e06 toy Word2Vec full-softmax results.", ANALYSIS_ROOT / "e06_word2vec_toy_full_softmax.png").parse_args()
    render(mlflow_client=client(args.tracking_uri), experiment_name=args.mlflow_experiment, atomic_run_ids=["W2V-TOY-CBOW-FULL", "W2V-TOY-SG-FULL"], metric="step/train/normalized_loss", title="e06 toy full-softmax Word2Vec", ylabel="normalized loss", output=args.output, summary_csv=args.summary_csv)


if __name__ == "__main__":
    main()
