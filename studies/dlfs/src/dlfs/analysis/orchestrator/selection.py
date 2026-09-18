from __future__ import annotations

from dlfs.definition import DEFINITION
from dlfs.identity import Variant, Volume
from repro_core.execution import RunOptions, RunSelection
from repro_core.execution.planning import Planner


def _canonical_device(
    volume: Volume,
    variant: Variant,
    *,
    study_id: str,
    condition_ids: tuple[str, ...],
) -> str | None:
    """Resolve analysis selection through the execution catalog's device policy."""
    if not condition_ids:
        return None
    plans = Planner(DEFINITION.implementation(volume, variant)).build(
        RunSelection(
            experiment_ids=(study_id,),
            atomic_run_ids=(condition_ids[0],),
        ),
        RunOptions(),
    )
    devices = {plan.device for plan in plans}
    if len(devices) != 1:
        raise ValueError(
            "analysis coordinate has multiple canonical devices: "
            f"{study_id}/{condition_ids[0]}: {sorted(devices)}"
        )
    return next(iter(devices))


def _seeds(selector, volume: Volume, study_id: str, condition, variants) -> set[str]:
    seeds = set()
    for variant in variants:
        aliases = set(condition.aliases(variant))
        seeds.update(
            item.seed
            for item in selector.attempts(volume, variant)
            if item.study_id == study_id
            and item.condition_id in aliases
            and item.status == "FINISHED"
            and item.disposition != "imported-alternate"
        )
    return seeds


def _seed_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)


def _row(study_id, condition_id, seed, metric, attempts, observations):
    protocols = {
        attempt.protocol_version for attempt in attempts.values() if attempt is not None
    }
    if protocols and protocols <= set(metric.protocols):
        protocol = metric.protocols[0]
    else:
        protocol = "" if not protocols else "protocol-mismatch"
    row = {
        "study_id": study_id,
        "experiment_id": study_id,
        "canonical_condition_id": condition_id,
        "condition_id": condition_id,
        "seed": seed,
        "metric_id": metric.metric_id,
        "unit": metric.unit,
        "split": metric.split,
        "axis": metric.axis,
        "protocol": protocol,
        "protocol_version": protocol,
    }
    for variant in Variant:
        attempt = attempts.get(variant)
        observation = observations.get(variant, {}).get(metric.metric_id)
        available = observation is not None and observation.available
        row.update(
            {
                f"{variant.value}_run_id": "" if attempt is None else attempt.run_id,
                f"{variant.value}_value": ""
                if not available
                else observation.values[-1],
                f"{variant.value}_availability": "available"
                if available
                else "unavailable",
                f"{variant.value}_unavailable_reason": (
                    "run is absent"
                    if attempt is None
                    else observation.unavailable_reason
                ),
                f"{variant.value}_native_schema": ""
                if observation is None
                else observation.native_schema,
                f"{variant.value}_provenance_ref": ""
                if observation is None
                else observation.provenance_ref,
            }
        )
    return row


OBSERVATION_CSV_FIELDS = (
    "study_id",
    "experiment_id",
    "canonical_condition_id",
    "condition_id",
    "seed",
    "metric_id",
    "unit",
    "split",
    "axis",
    "protocol",
    "protocol_version",
    "implemented_run_id",
    "implemented_value",
    "implemented_availability",
    "implemented_unavailable_reason",
    "implemented_native_schema",
    "implemented_provenance_ref",
    "original_run_id",
    "original_value",
    "original_availability",
    "original_unavailable_reason",
    "original_native_schema",
    "original_provenance_ref",
)


def _write_observations_csv(path, rows: list[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OBSERVATION_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
