"""DS2 data-loading helpers shared by individual analyses."""

from repro_core.analysis.core import aggregate


def runs(data, group: str, atomic_ids: list[str]):
    del group
    return data.runs(atomic_ids)


def source_curve(client, run_refs, metric: str):
    histories = client.histories_from_artifact(
        run_refs,
        artifact_path="observations/source_curves.csv",
        x="plot_index",
        y="value",
        row_filter=lambda row: row.get("metric") == metric,
    )
    if not histories and metric in {
        "book_loss",
        "loss",
        "perplexity",
        "exact_match_accuracy",
    }:
        # Promoted original DS2 runs retain the book's raw metrics.csv rather
        # than a canonical source_curves.csv. Project its native columns into
        # the same curve consumed by the shared renderer.
        y_column = {
            "book_loss": "loss",
            "loss": "loss",
            "perplexity": "perplexity",
            "exact_match_accuracy": "accuracy",
        }[metric]
        histories = client.histories_from_artifact(
            run_refs,
            artifact_path="raw/metrics.csv",
            x="plot_index" if metric != "exact_match_accuracy" else "epoch",
            y=y_column,
            row_filter=(
                (lambda row: row.get("split", "train") == "train")
                if metric == "perplexity"
                else None
            ),
        )
    return aggregate(histories)
