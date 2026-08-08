from __future__ import annotations

from copy import deepcopy

import pytest

from exp.ds2.profile.e02.plot_module_breakdown import (
    CONDITIONS,
    aggregate_module_breakdowns,
)


REGULAR_MEANS = {
    "batch_adapter": 0.1,
    "objective_prepare": 0.2,
    "model_forward": 0.3,
    "objective_forward": 0.4,
    "objective_backward": 0.5,
    "model_backward": 0.6,
    "optimizer": 0.7,
}
FUSED_MEANS = {
    "batch_adapter": 0.1,
    "objective_prepare": 0.2,
    "fused_forward_loss": 0.8,
    "fused_backward": 0.9,
    "optimizer": 0.7,
}


def _payloads() -> tuple[dict[str, object], dict[str, object]]:
    metadata = {
        "backend": "cupy",
        "device": "cuda:0",
        "cupy_version": "14.1.1",
    }
    update_rows = []
    module_rows = []
    for condition in CONDITIONS:
        model = "CBOW" if "-cbow-" in condition else "SkipGram"
        if condition.endswith("-fused-ns"):
            objective = "FusedNegativeSampling"
            components = FUSED_MEANS
            measurement_scope = "fused_negative_sampling"
        elif condition.endswith("-ns"):
            objective = "NegativeSampling"
            components = REGULAR_MEANS
            measurement_scope = "separate_model_objective"
        else:
            objective = "FullSoftmax"
            components = REGULAR_MEANS
            measurement_scope = "separate_model_objective"
        component_total = sum(components.values())
        update_rows.append(
            {
                "condition": condition,
                "implementation": "implemented",
                "model": model,
                "objective": objective,
                "batch_size": 100,
                "mean_ms_per_update": component_total + 0.25,
                "stdev_ms_per_update": 0.12,
            }
        )
        module_rows.extend(
            {
                "condition": condition,
                "implementation": "implemented",
                "model": model,
                "objective": objective,
                "component": component,
                "measurement_scope": measurement_scope,
                "batch_size": 100,
                "timing": {"mean_ms": mean},
            }
            for component, mean in components.items()
        )
    return (
        {"metadata": {**metadata, "stage": "update"}, "results": update_rows},
        {"metadata": {**metadata, "stage": "modules"}, "results": module_rows},
    )


def test_regular_negative_sampling_uses_separate_component_mapping() -> None:
    update_payload, modules_payload = _payloads()

    rows = aggregate_module_breakdowns(update_payload, modules_payload)
    row = next(
        row
        for row in rows
        if row.model == "CBOW" and row.method == "Negative Sampling"
    )

    assert row.prepare_ms == pytest.approx(0.3)
    assert row.forward_loss_ms == pytest.approx(0.7)
    assert row.backward_ms == pytest.approx(1.1)
    assert row.optimizer_ms == pytest.approx(0.7)
    assert row.other_ms == pytest.approx(0.25)
    assert row.stacked_total_ms == pytest.approx(row.update_mean_ms)


def test_fused_negative_sampling_uses_fused_component_mapping() -> None:
    update_payload, modules_payload = _payloads()

    rows = aggregate_module_breakdowns(update_payload, modules_payload)
    row = next(
        row
        for row in rows
        if row.model == "SkipGram" and row.method == "Fused Negative Sampling"
    )

    assert row.prepare_ms == pytest.approx(0.3)
    assert row.forward_loss_ms == pytest.approx(0.8)
    assert row.backward_ms == pytest.approx(0.9)
    assert row.optimizer_ms == pytest.approx(0.7)
    assert row.other_ms == pytest.approx(0.25)
    assert row.stacked_total_ms == pytest.approx(row.update_mean_ms)


def test_missing_component_identifies_condition_and_field() -> None:
    update_payload, modules_payload = _payloads()
    broken_modules = deepcopy(modules_payload)
    broken_modules["results"] = [
        row
        for row in broken_modules["results"]
        if not (
            row["condition"] == "implemented-cbow-ns"
            and row["component"] == "model_forward"
        )
    ]

    with pytest.raises(
        ValueError,
        match="implemented-cbow-ns.*model_forward",
    ):
        aggregate_module_breakdowns(update_payload, broken_modules)


def test_metadata_mismatch_is_rejected() -> None:
    update_payload, modules_payload = _payloads()
    modules_payload["metadata"]["device"] = "cuda:1"

    with pytest.raises(ValueError, match="metadata differ.*device"):
        aggregate_module_breakdowns(update_payload, modules_payload)
