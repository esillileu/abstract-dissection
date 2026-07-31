"""Profile implemented e02 Word2Vec modules independently.

Preparation needed to populate forward caches is executed outside the measured
region.  This makes objective/model backward timings independent while keeping
their inputs and tensor shapes identical to the real e02 update path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from mlprosection.profiling import BenchmarkRunner

from .update import (
    CONDITIONS,
    ROOT,
    _batch,
    _build_condition,
    _load_data,
    _metadata,
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
ORIGINAL_COMPONENTS = (
    "forward",
    "backward",
    "deduplicate_shared_parameters",
    "optimizer",
)
COMPONENTS = IMPLEMENTED_COMPONENTS + ORIGINAL_COMPONENTS
DEFAULT_OUTPUT = ROOT / "exp/ds2/profile/e02/results/modules.json"


class ComponentFixture:
    """Prepare real e02 tensors and caches for one measured component."""

    def __init__(self, workload, *, batch_size: int) -> None:
        self.workload = workload
        self.batch_x, self.batch_t = _batch(workload, 0, batch_size)
        self.model_x = None
        self.objective_t = None
        self.objective_batch = None
        self.prediction = None
        self.result = None
        self.gradient = None

    def batch_adapter(self) -> None:
        self.model_x, self.objective_t = self.workload.adapter.prepare(
            self.batch_x,
            self.batch_t,
        )

    def prepare_objective_prepare(self) -> None:
        self.batch_adapter()

    def objective_prepare(self) -> None:
        self.objective_batch = self.workload.objective.prepare(self.objective_t)

    def prepare_model_forward(self) -> None:
        self.batch_adapter()
        self.objective_prepare()

    def model_forward(self) -> None:
        self.prediction = self.workload.model.forward(
            self.model_x,
            candidates=self.objective_batch.candidates,
        )

    def prepare_objective_forward(self) -> None:
        self.prepare_model_forward()
        self.model_forward()

    def objective_forward(self) -> None:
        self.result = self.workload.objective.forward(
            self.prediction,
            self.objective_batch.target,
            replay_context=self.objective_batch.replay_context,
            example_count=len(self.batch_x),
        )

    def prepare_objective_backward(self) -> None:
        self.prepare_objective_forward()
        self.objective_forward()

    def objective_backward(self) -> None:
        self.gradient = self.workload.objective.backward()

    def prepare_model_backward(self) -> None:
        self.prepare_objective_backward()
        self.objective_backward()

    def model_backward(self) -> None:
        self.workload.model.backward(self.gradient)

    def prepare_optimizer(self) -> None:
        self.prepare_model_backward()
        self.model_backward()

    def optimizer(self) -> None:
        self.workload.optimizer.update()

    def operation(self, name: str):
        return getattr(self, name)

    def preparation(self, name: str):
        prepare = getattr(self, f"prepare_{name}", None)
        return prepare


class OriginalComponentFixture:
    """Expose the indivisible module boundaries of the book implementation."""

    def __init__(self, workload, *, batch_size: int) -> None:
        self.workload = workload
        self.batch_x, self.batch_t = _batch(workload, 0, batch_size)
        self.params = None
        self.grads = None

    def forward(self) -> None:
        self.workload.model.forward(self.batch_x, self.batch_t)

    def prepare_backward(self) -> None:
        self.forward()

    def backward(self) -> None:
        self.workload.model.backward()

    def prepare_deduplicate_shared_parameters(self) -> None:
        self.prepare_backward()
        self.backward()

    def deduplicate_shared_parameters(self) -> None:
        self.params, self.grads = self.workload.remove_duplicate(
            self.workload.model.params,
            self.workload.model.grads,
        )

    def prepare_optimizer(self) -> None:
        self.prepare_deduplicate_shared_parameters()
        self.deduplicate_shared_parameters()

    def optimizer(self) -> None:
        self.workload.optimizer.update(self.params, self.grads)

    def operation(self, name: str):
        return getattr(self, name)

    def preparation(self, name: str):
        return getattr(self, f"prepare_{name}", None)


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
    if implementation == "implemented":
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        choices=CONDITIONS,
        dest="conditions",
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=COMPONENTS,
        dest="components",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.measured_iterations) < 1 or args.warmup_iterations < 0:
        parser.error(
            "batch size and measured iterations must be positive; warmup must "
            "be non-negative"
        )

    backend, corpus, contexts, targets = _load_data(args.device)
    rows = []
    for condition in args.conditions or CONDITIONS:
        backend.seed(1)
        np.random.seed(1)
        condition_rows = profile_modules(
            condition,
            corpus=corpus,
            contexts=contexts,
            targets=targets,
            backend=backend,
            batch_size=args.batch_size,
            components=(
                tuple(args.components) if args.components is not None else None
            ),
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
        )
        rows.extend(condition_rows)
        for row in condition_rows:
            timing = row["timing"]
            assert isinstance(timing, dict)
            print(
                f"{condition} {row['component']}: "
                f"{float(timing['mean_ms']):.3f} ± "
                f"{float(timing['stdev_ms']):.3f} ms",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "metadata": _metadata(backend, stage="modules"),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
