"""Audit and repair MLflow condition-parent relationships."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

PARENT_TAGS = ("mlflow.parentRunId", "parent.mlflow_run_id")


def _runs(
    client: MlflowClient,
    experiment_id: str,
    *,
    active_only: bool = True,
) -> list[Any]:
    return list(
        client.search_runs(
            [experiment_id],
            run_view_type=(ViewType.ACTIVE_ONLY if active_only else ViewType.ALL),
            max_results=50_000,
        )
    )


def reconcile_parent_links(
    client: MlflowClient,
    experiment_id: str,
    *,
    group_keys: Iterable[str] | None = None,
    apply: bool = False,
) -> list[dict[str, Any]]:
    """Relink seed trials when their group has exactly one active parent.

    Ambiguous and unresolved groups are reported but never modified.
    """
    selected_keys = None if group_keys is None else set(group_keys)
    runs = _runs(client, experiment_id)
    all_runs = _runs(client, experiment_id, active_only=False)
    runs_by_id = {run.info.run_id: run for run in runs}
    all_runs_by_id = {run.info.run_id: run for run in all_runs}
    parents_by_group: dict[str, list[Any]] = defaultdict(list)
    for run in runs:
        tags = run.data.tags
        group_key = tags.get("condition.group.key")
        if tags.get("run.type") == "condition_parent" and group_key:
            parents_by_group[group_key].append(run)

    children_by_group: dict[str, list[Any]] = defaultdict(list)
    for run in runs:
        tags = run.data.tags
        group_key = tags.get("condition.group.key")
        if (
            tags.get("run.type") == "seed_trial"
            and group_key
            and (selected_keys is None or group_key in selected_keys)
        ):
            children_by_group[group_key].append(run)

    entries: list[dict[str, Any]] = []
    for group_key, children in children_by_group.items():
        candidates = parents_by_group.get(group_key, [])
        if candidates:
            continue
        referenced_ids = {
            child.data.tags.get(PARENT_TAGS[0])
            for child in children
            if child.data.tags.get(PARENT_TAGS[0])
            == child.data.tags.get(PARENT_TAGS[1])
        }
        deleted_parents = [
            all_runs_by_id[parent_id]
            for parent_id in referenced_ids
            if parent_id in all_runs_by_id
            and all_runs_by_id[parent_id].info.lifecycle_stage == "deleted"
            and all_runs_by_id[parent_id].data.tags.get("run.type")
            == "condition_parent"
            and all_runs_by_id[parent_id].data.tags.get("condition.group.key")
            == group_key
        ]
        if len(deleted_parents) != 1:
            continue
        parent = deleted_parents[0]
        entries.append(
            {
                "action": "restore-parent",
                "experiment_id": experiment_id,
                "condition_group_key": group_key,
                "parent_run_id": parent.info.run_id,
                "affected_child_run_ids": [child.info.run_id for child in children],
                "reason": "the uniquely referenced condition parent is soft-deleted",
            }
        )
        if apply:
            client.restore_run(parent.info.run_id)
        parents_by_group[group_key] = [parent]

    for group_key, children in children_by_group.items():
        candidates = parents_by_group.get(group_key, [])
        for run in children:
            current_ids = {tag: run.data.tags.get(tag) for tag in PARENT_TAGS}
            _reconcile_child(
                client,
                experiment_id,
                run,
                group_key,
                current_ids,
                candidates,
                runs_by_id,
                entries,
                apply=apply,
            )
    return entries


def _reconcile_child(
    client: MlflowClient,
    experiment_id: str,
    run: Any,
    group_key: str,
    current_ids: dict[str, str | None],
    candidates: list[Any],
    runs_by_id: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    apply: bool,
) -> None:
    """Reconcile one child after group-level parent restoration planning."""
    if len(candidates) == 1:
        expected_id = candidates[0].info.run_id
        changes = {
            tag: expected_id
            for tag, current_id in current_ids.items()
            if current_id != expected_id
        }
        if not changes:
            return
        entry = {
            "action": "relink",
            "experiment_id": experiment_id,
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "condition_group_key": group_key,
            "old_parent_ids": current_ids,
            "new_parent_id": expected_id,
            "reason": _repair_reason(current_ids, runs_by_id),
        }
        entries.append(entry)
        if apply:
            for tag, value in changes.items():
                client.set_tag(run.info.run_id, tag, value)
        return

    primary_id = current_ids[PARENT_TAGS[0]]
    if primary_id in runs_by_id and current_ids[PARENT_TAGS[1]] == primary_id:
        # The MLflow relationship itself is intact. A group-key mismatch can
        # originate from an older schema and is not safe to rewrite here.
        return
    entries.append(
        {
            "action": "unresolved" if not candidates else "ambiguous",
            "experiment_id": experiment_id,
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "condition_group_key": group_key,
            "old_parent_ids": current_ids,
            "candidate_parent_ids": [candidate.info.run_id for candidate in candidates],
            "reason": "no matching active parent"
            if not candidates
            else "multiple matching active parents",
        }
    )


def _repair_reason(
    current_ids: dict[str, str | None],
    runs_by_id: dict[str, Any],
) -> str:
    primary_id = current_ids[PARENT_TAGS[0]]
    alias_id = current_ids[PARENT_TAGS[1]]
    if not primary_id:
        return "missing MLflow parent tag"
    if primary_id not in runs_by_id:
        return "parent run is absent or inactive"
    if alias_id != primary_id:
        return "parent tag aliases disagree"
    return "parent belongs to a different condition group"


def relink_parents(
    tracking_uri: str,
    experiment_names: Iterable[str],
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Audit or repair parent links in the named experiments."""
    client = MlflowClient(tracking_uri=tracking_uri)
    entries: list[dict[str, Any]] = []
    for name in experiment_names:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            raise ValueError(f"MLflow experiment not found: {name}")
        entries.extend(
            reconcile_parent_links(
                client,
                experiment.experiment_id,
                apply=apply,
            )
        )
    return {
        "schema_version": 1,
        "command": "relink-parents",
        "mode": "apply" if apply else "dry-run",
        "created_at": datetime.now(UTC).isoformat(),
        "entries": entries,
    }
