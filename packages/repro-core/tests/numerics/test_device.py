import pytest

from repro_core.numerics import (
    DeviceMismatchError,
    assert_same_device,
    device_index,
    get_device,
    normalize_device,
    same_device,
)


class DummyDevice:
    def __init__(self, device: str) -> None:
        self.device = device


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cpu", "cpu"),
        ("CPU", "cpu"),
        (" gpu ", "cuda:0"),
        ("cuda:1", "cuda:1"),
        ("cuda0", "cuda:0"),
        ("cuda:01", "cuda:1"),
    ],
)
def test_normalize_device_valid(raw: str, expected: str) -> None:
    assert normalize_device(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "tpu",
        "gpu:0",
        "cuda:x",
    ],
)
def test_normalize_device_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_device(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cpu", None),
        ("gpu", 0),
        ("cuda:2", 2),
    ],
)
def test_device_index(raw: str, expected: int | None) -> None:
    assert device_index(raw) == expected


def test_get_device() -> None:
    assert get_device("cpu") == "cpu"
    assert get_device("cuda") == "cuda:0"
    assert get_device(DummyDevice("cuda1")) == "cuda:1"
    with pytest.raises(TypeError):
        get_device(object())


def test_same_device_and_assertions() -> None:
    assert same_device("cuda", "cuda:0", DummyDevice("cuda0"))
    assert not same_device("cpu", "cuda")

    assert_same_device("cuda", "cuda:0", DummyDevice("cuda0"))
    with pytest.raises(DeviceMismatchError):
        assert_same_device("cpu", "cuda")
