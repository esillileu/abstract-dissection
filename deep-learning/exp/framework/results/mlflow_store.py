"""MLflow-backed loading for domain-declared native metrics."""

from __future__ import annotations

from collections.abc import Iterable

from .store import ArtifactReference, MetricSeries, NativeRunResult


class MlflowResultStore:
    """Load raw run data without interpreting experiment vocabulary."""

    def __init__(self, client) -> None:
        self.client = client

    def list_run_ids(self, **selection: object) -> tuple[str, ...]:
        experiment_ids = selection.get("experiment_ids", ())
        filter_string = str(selection.get("filter_string", ""))
        runs = self.client.search_runs(
            list(experiment_ids),
            filter_string=filter_string,
            order_by=["attributes.start_time DESC"],
            max_results=int(selection.get("max_results", 10_000)),
        )
        return tuple(run.info.run_id for run in runs)

    def load(
        self,
        run_id: str,
        *,
        metric_specs: Iterable[tuple[str, str, str, str]],
        include_artifacts: bool = True,
    ) -> NativeRunResult:
        run = self.client.get_run(run_id)
        metrics = []
        for metric_id, unit, split, axis in metric_specs:
            history = tuple(self.client.get_metric_history(run_id, metric_id))
            if not history and metric_id in run.data.metrics:
                metrics.append(MetricSeries(
                    metric_id, unit, split, axis, (0,),
                    (float(run.data.metrics[metric_id]),),
                ))
                continue
            if history:
                ordered = sorted(history, key=lambda item: (item.step, item.timestamp))
                metrics.append(MetricSeries(
                    metric_id, unit, split, axis,
                    tuple(int(item.step) for item in ordered),
                    tuple(float(item.value) for item in ordered),
                ))
        artifacts = tuple(self._artifacts(run_id)) if include_artifacts else ()
        tags = run.data.tags
        schema_version = tags.get("result.schema.version", tags.get("schema.version", "0"))
        try:
            parsed_schema_version = int(schema_version or 0)
        except ValueError:
            parsed_schema_version = 0
        return NativeRunResult(
            run_id=run_id,
            schema_name=tags.get("result.schema.name", tags.get("schema.version", "legacy")),
            schema_version=parsed_schema_version,
            protocol_version=tags.get("protocol.version", "legacy"),
            metrics=tuple(metrics),
            artifacts=artifacts,
            provenance={
                "experiment_id": tags.get("experiment.id"),
                "namespace": tags.get("transfer.source.experiment_name"),
                "source_run_id": tags.get("transfer.source.run_id", run_id),
                "payload_sha256": tags.get("transfer.payload.sha256"),
                "git_commit": tags.get("code.git_commit"),
            },
            provenance_ref=tags.get("provenance.ref") or tags.get("transfer.source.run_id"),
        )

    def _artifacts(self, run_id: str) -> list[ArtifactReference]:
        output: list[ArtifactReference] = []
        pending = [""]
        while pending:
            prefix = pending.pop()
            try:
                entries = self.client.list_artifacts(run_id, prefix)
            except Exception:
                break
            for entry in entries:
                if entry.is_dir:
                    pending.append(entry.path)
                else:
                    output.append(ArtifactReference(entry.path, size=entry.file_size))
        return output
