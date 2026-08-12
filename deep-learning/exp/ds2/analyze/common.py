"""DS2 data-loading helpers shared by individual analyses."""

from exp.analyze import aggregate, completed_seed_runs, histories_from_artifact


BOOK_SOURCE_GROUPS = {"GT03", "GT04", "GT05"}


def runs(client, group: str, atomic_ids: list[str]):
    return completed_seed_runs(
        client,
        experiment_name="ds2",
        group_id=group,
        atomic_run_ids=atomic_ids,
        protocol_version=(
            "book-source-v1" if group in BOOK_SOURCE_GROUPS else "legacy"
        ),
    )


def source_curve(client, run_refs, metric: str):
    return aggregate(
        histories_from_artifact(
            client,
            run_refs,
            artifact_path="observations/source_curves.csv",
            x="plot_index",
            y="value",
            row_filter=lambda row: row.get("metric") == metric,
        )
    )
