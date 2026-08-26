"""DS1 data-loading helpers shared by individual analyses."""

from __future__ import annotations

from repro_core.analysis.core import Curve, aggregate, smooth_histories


def runs(data, group: str, atomic_ids: list[str]):
    del group
    return data.runs(atomic_ids)


def loss_curve(client, run_refs):
    histories = []
    for run in run_refs:
        series = run.result.metric("train/loss") or run.result.metric("train/objective")
        if series is not None:
            histories.append(
                {
                    float(step) - 1: float(value)
                    for step, value in zip(series.steps, series.values, strict=True)
                }
            )
    if not histories:
        histories = client.histories_from_artifact(
            run_refs,
            artifact_path="updates.csv",
            x="update",
            y="loss",
            x_value=lambda row: float(row["update"]) - 1,
        )
    return aggregate(smooth_histories(histories))


def accuracy_curve(client, run_refs, *, split: str, x_value, axis: str | None = None):
    histories = client.histories_from_artifact(
        run_refs,
        artifact_path="evaluations.csv",
        x="axis_step",
        y="accuracy",
        row_filter=lambda row: (
            row.get("split") == split and (axis is None or row.get("axis") == axis)
        ),
        x_value=x_value,
    )
    if not histories:
        # Promoted original DS1 runs keep the source evaluation rows in
        # raw/metrics.csv.  Prefer that artifact over the persisted MLflow
        # metric series, whose step is the global update counter.
        histories = client.histories_from_artifact(
            run_refs,
            artifact_path="raw/metrics.csv",
            x="epoch",
            y="accuracy",
            row_filter=lambda row: row.get("split") == split,
            x_value=x_value,
        )
    if not histories:
        native_ids = (
            ("update/eval_train/accuracy", "train/accuracy")
            if split == "train"
            else ("update/eval_test/accuracy", "test/accuracy")
        )
        for run in run_refs:
            series = next(
                (
                    run.result.metric(metric_id)
                    for metric_id in native_ids
                    if run.result.metric(metric_id) is not None
                ),
                None,
            )
            if series is not None:
                histories.append(dict(zip(series.steps, series.values, strict=True)))
    return aggregate(histories)


def accuracy_percent_curve(
    client, run_refs, *, split: str, x_value, axis: str | None = None
):
    curve = accuracy_curve(client, run_refs, split=split, x_value=x_value, axis=axis)
    return Curve(
        steps=curve.steps,
        mean=curve.mean * 100.0,
        minimum=curve.minimum * 100.0,
        maximum=curve.maximum * 100.0,
        run_count=curve.run_count,
        standard_deviation=None
        if curve.standard_deviation is None
        else curve.standard_deviation * 100.0,
    )
