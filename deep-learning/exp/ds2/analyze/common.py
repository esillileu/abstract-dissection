"""DS2 data-loading helpers shared by individual analyses."""

from exp.analyze import aggregate, completed_seed_runs, histories_from_artifact


BOOK_SOURCE_GROUPS = {"GT03", "GT04", "GT05", "GT06", "GT07", "GT09", "GO01"}


def runs(client, group: str, atomic_ids: list[str]):
    legacy = getattr(client, "analysis_legacy", False)
    protocol_version = (
        "legacy"
        if legacy
        else "book-source-v1" if group in BOOK_SOURCE_GROUPS else "legacy"
    )
    grouped = completed_seed_runs(
        client,
        experiment_name="ds2",
        group_id=group,
        atomic_run_ids=atomic_ids,
        protocol_version=protocol_version,
    )
    if legacy:
        return grouped

    missing = [atomic_id for atomic_id in atomic_ids if not grouped[atomic_id]]
    if not missing:
        return grouped
    fallback = completed_seed_runs(
        client,
        experiment_name="ds2",
        group_id=group,
        atomic_run_ids=missing,
        protocol_version=None,
    )
    for atomic_id in missing:
        grouped[atomic_id] = fallback[atomic_id]
    return grouped


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
