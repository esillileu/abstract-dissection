from .common import ANALYSIS_ROOT, client, parser
from .render import render


EXPERIMENT_ID = "e03"


def main() -> None:
    args = parser("Render e03 language-model perplexity.", ANALYSIS_ROOT / "e03_rnnlm_comparison.png").parse_args()
    render(mlflow_client=client(args.tracking_uri), experiment_name=args.mlflow_experiment, atomic_run_ids=["LM-RNN-C025", "LM-LSTM-C025", "LM-BETTER"], metric="step/train/perplexity", title="e03 train perplexity", ylabel="perplexity", output=args.output, summary_csv=args.summary_csv)


if __name__ == "__main__":
    main()
