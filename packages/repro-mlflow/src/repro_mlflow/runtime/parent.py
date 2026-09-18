"""MLflow condition parent run resolution and tracking hierarchy."""

from __future__ import annotations


def get_or_create_condition_parent(
    client, *, experiment_id: str, child_tags: dict[str, str]
) -> str:
    """Return the condition parent shared by all seed trials of one condition."""
    condition_key = child_tags.get("condition.key")
    group_key = child_tags.get("condition.group.key", condition_key)
    if not group_key:
        raise ValueError("seed trial tags require condition.group.key or condition.key")
    filter_string = (
        "tags.`run.type` = 'condition_parent' "
        f"AND tags.`condition.group.key` = '{group_key}'"
    )
    parents = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time ASC"],
        max_results=1,
    )
    if parents:
        return parents[0].info.run_id

    legacy_parent = _find_legacy_condition_parent(
        client,
        experiment_id=experiment_id,
        child_tags=child_tags,
    )
    if legacy_parent is not None:
        client.set_tag(legacy_parent.info.run_id, "condition.group.key", group_key)
        return legacy_parent.info.run_id

    parent_tags = {
        key: value
        for key, value in child_tags.items()
        if key
        not in {
            "run.key",
            "master_seed",
            "trial.status",
            "trial.attempt",
            "retry.of",
            "parent.mlflow_run_id",
            "mlflow.parentRunId",
        }
    }
    parent_tags.update(
        {
            "run.type": "condition_parent",
            "condition.status": "running",
            "condition.group.key": group_key,
            "mlflow.runName": child_tags.get(
                "atomic_run.id", f"condition-{condition_key[:12]}"
            ),
        }
    )
    parent = client.create_run(experiment_id=experiment_id, tags=parent_tags)
    client.set_terminated(parent.info.run_id, status="FINISHED")
    return parent.info.run_id


def _find_legacy_condition_parent(
    client, *, experiment_id: str, child_tags: dict[str, str]
):
    """Find a pre-canonical-key parent using the declared experiment identity."""
    identity_tags = (
        "experiment.ids",
        "execution_group.id",
        "recipe.id",
        "structure.signature",
        "atomic_run.id",
    )
    if any(not child_tags.get(key) for key in identity_tags):
        return None
    parents = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.`run.type` = 'condition_parent'",
        order_by=["attributes.start_time ASC"],
        max_results=10_000,
    )
    for parent in parents:
        parent_tags = parent.data.tags
        if all(parent_tags.get(key) == child_tags[key] for key in identity_tags):
            return parent
    return None


__all__ = [
    "get_or_create_condition_parent",
]
