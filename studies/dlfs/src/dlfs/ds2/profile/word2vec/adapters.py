"""Typed Word2Vec workload adapters for canonical profile studies."""

from __future__ import annotations

import gc

import numpy as np

from dlfs.profile import ProfileSection

from .contracts import Word2VecCondition


class UpdateWorkloadAdapter:
    def __init__(self, workload, *, backend, batch_size: int) -> None:
        self._workload = workload
        self.backend = backend
        self.batch_size = batch_size
        self.next_index = 0

    def update(self) -> None:
        from .workloads import profile_batch

        batch_x, batch_t = profile_batch(
            self._workload, self.next_index, self.batch_size
        )
        self._workload.update(batch_x, batch_t)
        self.next_index += 1

    def sections(self):
        return {}

    def metadata(self):
        return {"batch_size": self.batch_size}

    def release(self) -> None:
        self._workload = None
        _release_backend(self.backend)


class ScalingWorkloadAdapter:
    def __init__(self, workload, *, backend, batch_size: int) -> None:
        self._workload = workload
        self.backend = backend
        self.batch_size = batch_size
        self.next_index = 0

    def update(self) -> None:
        self._workload.update(self.next_index, self.batch_size)
        self.next_index += 1

    def sections(self):
        return {}

    def metadata(self):
        return {"batch_size": self.batch_size}

    def release(self) -> None:
        self._workload = None
        _release_backend(self.backend)


class ModuleWorkloadAdapter:
    def __init__(
        self,
        workload,
        fixture,
        components,
        *,
        backend,
        measurement_scope: str,
    ) -> None:
        self._workload = workload
        self._fixture = fixture
        self._components = components
        self.backend = backend
        self.measurement_scope = measurement_scope

    def update(self) -> None:
        raise RuntimeError("module workload exposes sections, not update")

    def sections(self):
        return {
            component: ProfileSection(
                operation=self._fixture.operation(component),
                prepare=self._fixture.preparation(component),
            )
            for component in self._components
        }

    def metadata(self):
        return {"measurement_scope": self.measurement_scope}

    def release(self) -> None:
        self._fixture = None
        self._workload = None
        _release_backend(self.backend)


def build_update_workload(
    condition: Word2VecCondition, *, device: str, batch_size: int
):
    from .workloads import (
        build_profile_condition,
        load_profile_data,
        profile_metadata,
    )

    backend, corpus, contexts, targets = load_profile_data(device)
    workload, model_name, objective_name, implementation = build_profile_condition(
        condition.legacy_id(),
        corpus=corpus,
        contexts=contexts,
        targets=targets,
        backend=backend,
    )
    adapter = UpdateWorkloadAdapter(workload, backend=backend, batch_size=batch_size)
    source = {
        "corpus": corpus,
        "contexts": contexts,
        "targets": targets,
        "backend": backend,
        "dataset_samples": len(contexts),
        "model": model_name,
        "objective": objective_name,
        "implementation": implementation,
    }
    return adapter, source, profile_metadata(backend, stage="canonical")


def build_module_workload(
    condition: Word2VecCondition,
    *,
    source: dict[str, object],
    batch_size: int,
):
    from .sections import (
        FUSED_COMPONENTS,
        IMPLEMENTED_COMPONENTS,
        ORIGINAL_COMPONENTS,
        ComponentFixture,
        FusedComponentFixture,
        OriginalComponentFixture,
    )
    from .workloads import build_profile_condition

    workload, _model, _objective, implementation = build_profile_condition(
        condition.legacy_id(),
        corpus=source["corpus"],
        contexts=source["contexts"],
        targets=source["targets"],
        backend=source["backend"],
    )
    if implementation == "implemented" and workload.fused:
        fixture = FusedComponentFixture(workload, batch_size=batch_size)
        components = FUSED_COMPONENTS
        scope = "fused_negative_sampling"
    elif implementation == "implemented":
        fixture = ComponentFixture(workload, batch_size=batch_size)
        components = IMPLEMENTED_COMPONENTS
        scope = "separate_model_objective"
    else:
        fixture = OriginalComponentFixture(workload, batch_size=batch_size)
        components = ORIGINAL_COMPONENTS
        scope = "combined_model_objective"
    return ModuleWorkloadAdapter(
        workload,
        fixture,
        components,
        backend=source["backend"],
        measurement_scope=scope,
    )


def build_scaling_workload(
    condition: Word2VecCondition,
    *,
    vocabulary_size: int,
    backend,
    batch_size: int,
    update_count: int,
) -> ScalingWorkloadAdapter:
    from .scaling import (
        ScalingWorkload,
        synthetic_scaling_batches,
    )

    contexts, targets = synthetic_scaling_batches(
        vocabulary_size,
        batch_size=batch_size,
        update_count=update_count,
    )
    backend.seed(1)
    np.random.seed(1)
    workload = ScalingWorkload(
        condition.legacy_id(),
        vocab_size=vocabulary_size,
        contexts=contexts,
        targets=targets,
        backend=backend,
    )
    return ScalingWorkloadAdapter(workload, backend=backend, batch_size=batch_size)


def _release_backend(backend) -> None:
    gc.collect()
    if backend.is_gpu:
        backend.synchronize()
        backend.xp.get_default_memory_pool().free_all_blocks()
        backend.xp.get_default_pinned_memory_pool().free_all_blocks()


def make_scaling_backend(*, device: str, dtype: str, seed: int):
    from deepscratch.core import BackendConfig, make_backend

    return make_backend(
        BackendConfig(
            device=device,
            dtype=dtype,
            seed=seed,
            profile=device.startswith("cuda:"),
        )
    )


def default_vocabulary_sizes(device: str) -> tuple[int, ...]:
    from .scaling import (
        default_vocabulary_sizes,
    )

    return default_vocabulary_sizes(device)


def scaling_environment(backend) -> dict[str, object]:
    from .workloads import profile_metadata

    return profile_metadata(backend, stage="vocabulary_size_scaling")


def scaling_crossovers(rows: list[dict[str, object]]) -> dict[str, object]:
    from .scaling import (
        summarize_crossovers,
    )

    return summarize_crossovers(rows)
