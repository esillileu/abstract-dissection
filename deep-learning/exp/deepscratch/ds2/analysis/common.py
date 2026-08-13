"""DS2 data-loading helpers shared by individual analyses."""

from exp.framework.analysis.core import aggregate


def runs(data, group: str, atomic_ids: list[str]):
    del group
    return data.runs(atomic_ids)


def source_curve(client, run_refs, metric: str):
    return aggregate(
        client.histories_from_artifact(
            run_refs,
            artifact_path="observations/source_curves.csv",
            x="plot_index",
            y="value",
            row_filter=lambda row: row.get("metric") == metric,
        )
    )
