from __future__ import annotations

from contextlib import nullcontext

from exp.deepscratch.profile import MeasurementProtocol, ProfileSection, ScalingAxis
from exp.deepscratch.profile.engine import (
    measure_update_workload,
    measure_workload_sections,
)
from exp.deepscratch.ds2.profile.word2vec import Word2VecCondition


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
    condition = Word2VecCondition.from_mapping({
        "subject_variant": "implemented",
        "model": "skipgram",
        "objective": "fused_negative_sampling",
    })
    assert condition.legacy_id() == "implemented-skipgram-fused-ns"
    axis = ScalingAxis.from_mapping({
        "name": "batch_size",
        "values": [32, 64],
        "reverse": True,
    })
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
