from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from .common import ANALYSIS_ROOT, client, latest_seeded_runs, parser
from .render import render


def main() -> None:
    args = parser("Render e13 date exact-match accuracy.", ANALYSIS_ROOT / "e13_attention_seq2seq_date.png").parse_args()
    mlflow_client = client(args.tracking_uri)
    render(mlflow_client=mlflow_client, experiment_name=args.mlflow_experiment, atomic_run_ids=["SEQD-VAN-REV", "SEQD-PEEKY-REV", "SEQD-ATTN-REV"], metric="epoch/test/exact_match", title="e13 date exact match", ylabel="exact-match accuracy", output=args.output, summary_csv=args.summary_csv, marker="o")
    _render_attention(mlflow_client, args.mlflow_experiment, ANALYSIS_ROOT / "e13_attention_alignment.png")


def _render_attention(mlflow_client, experiment_name: str, output: Path) -> None:
    run = latest_seeded_runs(mlflow_client, experiment_name, ["SEQD-ATTN-REV"])["SEQD-ATTN-REV"][0]
    with TemporaryDirectory() as temporary:
        artifact = mlflow_client.download_artifacts(run.info.run_id, "analysis/attention_map.npz", temporary)
        with np.load(artifact) as values:
            attention = values["attention"]
        figure, axis = plt.subplots(figsize=(7, 4))
        image = axis.pcolor(attention, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)
        axis.set_xlabel("encoder character position")
        axis.set_ylabel("decoder character position")
        axis.set_title("e13 representative attention alignment")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main()
