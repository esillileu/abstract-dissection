from __future__ import annotations

from contextlib import nullcontext

import numpy as np
from deepscratch.core import BackendConfig, make_backend

from dlfs.ds2.profile.word2vec import Word2VecCondition
from dlfs.ds2.profile.word2vec.workloads import _build_condition
from dlfs.profile import MeasurementProtocol, ProfileSection, ScalingAxis
from dlfs.profile.engine import (
    measure_update_workload,
    measure_workload_sections,
)


class _Backend:
    is_gpu = False

    def synchronize(self) -> None:
        pass

    def range(self, _name: str):
        return nullcontext()


class _Workload:
    def __init__(self) -> None:
        self.backend = _Backend()
        self.updates = 0
        self.released = False

    def update(self) -> None:
        self.updates += 1

    def sections(self):
        return {"update": ProfileSection(self.update)}

    def metadata(self):
        return {"kind": "test"}

    def release(self) -> None:
        self.released = True


def test_generic_profile_engine_measures_and_releases_workload() -> None:
    workload = _Workload()
    point = measure_update_workload(
        "condition",
        workload,
        axes={"width": 16},
        protocol=MeasurementProtocol(1, 2, 2),
    )
    assert point.status == "ok"
    assert point.axes == {"width": 16}
    assert point.metrics["update_ms"] is not None
    assert workload.updates == 1 + 1 + 2 + 2 * 2
    assert workload.released


def test_typed_condition_and_generic_axis_are_validated() -> None:
    condition = Word2VecCondition.from_mapping(
        {
            "subject_variant": "implemented",
            "model": "skipgram",
            "objective": "fused_negative_sampling",
        }
    )
    assert condition.legacy_id() == "implemented-skipgram-fused-ns"
    axis = ScalingAxis.from_mapping(
        {
            "name": "batch_size",
            "values": [32, 64],
            "reverse": True,
        }
    )
    assert axis.name == "batch_size"
    assert axis.values == (32, 64)
    assert axis.reverse


def test_generic_profile_engine_measures_declared_sections() -> None:
    workload = _Workload()
    sections = measure_workload_sections(
        workload, warmup_iterations=1, measured_iterations=2
    )
    assert sections["update"]["timing"]["count"] == 2
    assert workload.updates == 3
    assert workload.released


def test_one_hot_skipgram_full_softmax_accepts_grouped_context_targets() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32", seed=1))
    corpus = np.arange(8, dtype=np.int64)
    contexts = np.array([[0, 2], [1, 3], [2, 4], [3, 5]], dtype=np.int64)
    targets = np.array([1, 2, 3, 4], dtype=np.int64)
    workload, _, _, _ = _build_condition(
        "implemented-skipgram-onehot-fs",
        corpus=corpus,
        contexts=contexts,
        targets=targets,
        backend=backend,
    )

    loss = workload.update(workload.contexts[:2], workload.targets[:2])

    assert workload.objective.grouped_targets is True
    assert np.isfinite(float(loss.data))
