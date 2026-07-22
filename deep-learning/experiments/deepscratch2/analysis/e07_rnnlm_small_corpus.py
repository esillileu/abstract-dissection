from .common import ANALYSIS_ROOT, client, parser
from .render import render


EXPERIMENT_ID = "e07"


def main() -> None:
    args = parser("Render e07 small-corpus RNNLM results.", ANALYSIS_ROOT / "e07_rnnlm_small_corpus.png").parse_args()
    render(mlflow_client=client(args.tracking_uri), experiment_name=args.mlflow_experiment, atomic_run_ids=["LM-TOY-RNN", "LM-TOY-LSTM", "LM-TOY-BETTER"], metric="step/train/perplexity", title="e07 small-corpus train perplexity", ylabel="perplexity", output=args.output, summary_csv=args.summary_csv)


if __name__ == "__main__":
    main()
