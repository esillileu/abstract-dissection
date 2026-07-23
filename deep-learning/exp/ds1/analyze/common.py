"""DS1 data-loading helpers shared by individual analyses."""

from __future__ import annotations

from exp.analyze import aggregate, completed_seed_runs, histories_from_artifact, smooth_histories


def runs(client, group: str, atomic_ids: list[str]):
    return completed_seed_runs(
        client,
        experiment_name="ds1",
        group_id=group,
        atomic_run_ids=atomic_ids,
    )


def loss_curve(client, run_refs):
    histories = histories_from_artifact(
        client,
        run_refs,
        artifact_path="updates.csv",
        x="update",
        y="loss",
        x_value=lambda row: float(row["update"]) - 1,
    )
    return aggregate(smooth_histories(histories))


def accuracy_curve(client, run_refs, *, split: str, x_value, axis: str | None = None):
    return aggregate(
        histories_from_artifact(
            client,
            run_refs,
            artifact_path="evaluations.csv",
            x="axis_step",
            y="accuracy",
            row_filter=lambda row: row.get("split") == split
            and (axis is None or row.get("axis") == axis),
            x_value=x_value,
        )
    )
