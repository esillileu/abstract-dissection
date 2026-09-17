from __future__ import annotations

from collections.abc import Callable

from dlfs.ds1.implemented.final_gap import (
    TRAIN_FULL_ACCURACY,
    TRAIN_TEST_ACCURACY_GAP,
)

TARGET_NAMES = {
    "e06": ("GT06", "CNN-SIMPLE-BOOK"),
    "e07": ("GT07", "CNN-DEEP-BOOK"),
    "e12": ("GT09", "MLP-EXT-ALL-BOOK"),
}


def latest_target_runs(client, *, targets: set[tuple[str, str]], seeds: set[int]):
    experiment = client.get_experiment_by_name("ds1")
    if experiment is None:
        raise ValueError("MLflow experiment does not exist: ds1")
    selected = {}
    for group_id, atomic_run_id in sorted(targets):
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED' and "
                "tags.`run.type` = 'seed_trial' and "
                f"tags.`execution_group.id` = '{group_id}' and "
                f"tags.`atomic_run.id` = '{atomic_run_id}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=5_000,
        )
        for run in runs:
            seed = int(
                run.data.params.get(
                    "seed/master",
                    run.data.params.get("seed", run.data.tags.get("master_seed", -1)),
                )
            )
            if seeds and seed not in seeds:
                continue
            selected.setdefault((group_id, atomic_run_id, seed), run)
    return list(selected.values())


def backfill_runs(
    client,
    runs,
    *,
    apply: bool,
    force: bool,
    evaluator: Callable[[object], dict[str, float]],
) -> dict[str, int]:
    counts = {"planned": 0, "updated": 0, "skipped": 0, "failed": 0}
    for run in runs:
        run_id = run.info.run_id
        atomic = run.data.tags.get("atomic_run.id", "")
        seed = run.data.params.get("seed/master", run.data.tags.get("master_seed", ""))
        present = {
            TRAIN_FULL_ACCURACY,
            TRAIN_TEST_ACCURACY_GAP,
        }.issubset(run.data.metrics)
        if present and not force:
            counts["skipped"] += 1
            print(f"skip {atomic} seed={seed} run={run_id}: metrics already exist")
            continue
        counts["planned"] += 1
        if not apply:
            print(f"would update {atomic} seed={seed} run={run_id}")
            continue
        try:
            metrics = evaluator(run)
            step = int(run.data.metrics.get("final/system/total_updates", 0))
            for key in (TRAIN_FULL_ACCURACY, TRAIN_TEST_ACCURACY_GAP):
                client.log_metric(run_id, key, float(metrics[key]), step=step)
            client.set_tag(run_id, "maintenance.full_train_gap", "checkpoint-v1")
        except Exception as exc:
            counts["failed"] += 1
            print(f"failed {atomic} seed={seed} run={run_id}: {exc}")
            continue
        counts["updated"] += 1
        print(
            f"updated {atomic} seed={seed} run={run_id}: "
            f"train_full={metrics[TRAIN_FULL_ACCURACY]:.6f} "
            f"gap={metrics[TRAIN_TEST_ACCURACY_GAP]:.6f}"
        )
    return counts
