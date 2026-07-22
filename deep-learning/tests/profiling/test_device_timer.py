from __future__ import annotations

from mlprosection.profiling.backend import CuPyDeviceTimer, NullDeviceTimer, create_device_timer


def test_null_device_timer_never_reports_elapsed_time() -> None:
    timer = NullDeviceTimer()

    token = timer.start()
    timer.stop(token)

    assert token is None
    assert timer.elapsed_ns(token) is None


def test_create_device_timer_uses_null_timer_when_disabled() -> None:
    backend = type("Backend", (), {"name": "cupy", "is_gpu": True, "xp": object()})()

    assert isinstance(create_device_timer(backend, enabled=False), NullDeviceTimer)


def test_create_device_timer_uses_null_timer_for_cpu_even_when_enabled() -> None:
    backend = type("Backend", (), {"name": "numpy", "is_gpu": False, "xp": object()})()

    assert isinstance(create_device_timer(backend, enabled=True), NullDeviceTimer)


def test_fake_cupy_device_timer_records_and_synchronizes_once() -> None:
    class FakeEvent:
        def __init__(self, cupy) -> None:
            self.cupy = cupy

        def record(self) -> None:
            self.cupy.records += 1

        def synchronize(self) -> None:
            self.cupy.synchronizes += 1

    class FakeCuda:
        def __init__(self, cupy) -> None:
            self.cupy = cupy

        def Event(self) -> FakeEvent:
            return FakeEvent(self.cupy)

        def get_elapsed_time(self, _start, _end) -> float:
            self.cupy.elapsed_calls += 1
            return 1.25

    class FakeCuPy:
        def __init__(self) -> None:
            self.records = 0
            self.synchronizes = 0
            self.elapsed_calls = 0
            self.cuda = FakeCuda(self)

    cp = FakeCuPy()
    timer = CuPyDeviceTimer(cp)

    token = timer.start()
    timer.stop(token)

    assert cp.records == 2
    assert cp.synchronizes == 0
    assert timer.elapsed_ns(token) == 1_250_000
    assert cp.synchronizes == 1
    assert cp.elapsed_calls == 1
