from .common import ANALYSIS_ROOT, client, parser
from .render import render


EXPERIMENT_ID = "e12"


def main() -> None:
    args = parser("Render e12 addition exact-match accuracy.", ANALYSIS_ROOT / "e12_seq2seq_addition.png").parse_args()
    render(mlflow_client=client(args.tracking_uri), experiment_name=args.mlflow_experiment, atomic_run_ids=["SEQA-VAN-FWD", "SEQA-VAN-REV", "SEQA-PEEKY-FWD", "SEQA-PEEKY-REV", "SEQA-ATTN-REV"], metric="epoch/test/exact_match", title="e12 addition exact match", ylabel="exact-match accuracy", output=args.output, summary_csv=args.summary_csv, marker="o")


if __name__ == "__main__":
    main()
