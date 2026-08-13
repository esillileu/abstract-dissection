"""DS2 data-loading helpers shared by individual analyses."""

from exp.framework.analysis.core import aggregate


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
    if not histories and metric == "book_loss":
        # Promoted original DS2 runs retain the book's raw metrics.csv rather
        # than a canonical source_curves.csv.  Its loss column is the same
        # interval curve and must remain available to the original renderer.
        histories = client.histories_from_artifact(
            run_refs,
            artifact_path="raw/metrics.csv",
            x="plot_index",
            y="loss",
        )
    return aggregate(histories)
