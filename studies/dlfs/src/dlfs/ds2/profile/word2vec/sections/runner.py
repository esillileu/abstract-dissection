"""Profile individual Word2Vec modules and components."""

from __future__ import annotations

from dataclasses import asdict

from deepscratch.profiling import BenchmarkRunner

from dlfs.ds2.profile.paths import profile_measurements

from ..workloads import _build_condition
from .fixtures import (
    ComponentFixture,
    FusedComponentFixture,
    OriginalComponentFixture,
)

IMPLEMENTED_COMPONENTS = (
    "batch_adapter",
    "objective_prepare",
    "model_forward",
    "objective_forward",
    "objective_backward",
    "model_backward",
    "optimizer",
)
FUSED_COMPONENTS = (
    "batch_adapter",
    "objective_prepare",
    "fused_forward_loss",
    "fused_backward",
    "optimizer",
)
ORIGINAL_COMPONENTS = (
    "forward",
    "backward",
    "deduplicate_shared_parameters",
    "optimizer",
)
COMPONENTS = IMPLEMENTED_COMPONENTS + FUSED_COMPONENTS + ORIGINAL_COMPONENTS
DEFAULT_OUTPUT = profile_measurements("e10") / "modules.json"


def profile_modules(
    condition: str,
    *,
    corpus,
    contexts,
    targets,
    backend,
    batch_size: int,
    components: tuple[str, ...] | None,
    warmup_iterations: int,
    measured_iterations: int,
) -> list[dict[str, object]]:
    workload, model_name, objective_name, implementation = _build_condition(
        condition,
        corpus=corpus,
        contexts=contexts,
        targets=targets,
        backend=backend,
    )
    if implementation == "implemented" and workload.fused:
        fixture = FusedComponentFixture(workload, batch_size=batch_size)
        available_components = FUSED_COMPONENTS
        measurement_scope = "fused_negative_sampling"
    elif implementation == "implemented":
        fixture = ComponentFixture(workload, batch_size=batch_size)
        available_components = IMPLEMENTED_COMPONENTS
        measurement_scope = "separate_model_objective"
    else:
        fixture = OriginalComponentFixture(workload, batch_size=batch_size)
        available_components = ORIGINAL_COMPONENTS
        measurement_scope = "combined_model_objective"
    selected_components = (
        available_components
        if components is None
        else tuple(
            component for component in components if component in available_components
        )
    )
    runner = BenchmarkRunner(backend)
    rows = []
    for component in selected_components:
        result = runner.measure_iterations(
            f"{condition}.{component}",
            fixture.operation(component),
            prepare=fixture.preparation(component),
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
        )
        row = {
            "condition": condition,
            "implementation": implementation,
            "model": model_name,
            "objective": objective_name,
            "component": component,
            "measurement_scope": measurement_scope,
            "batch_size": batch_size,
            **asdict(result),
        }
        rows.append(row)
    return rows
