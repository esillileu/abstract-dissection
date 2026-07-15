import pytest

from mlprosection.core.backend import (
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
        ("cuda", "cuda:0"),
        ("CUDA", "cuda:0"),
        ("cuda:0", "cuda:0"),
        ("cuda:1", "cuda:1"),
        ("cuda0", "cuda:0"),
        ("cuda1", "cuda:1"),
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
        "cuda:",
        "cuda:x",
        "cuda:x0",
        "cuda-0",
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
        ("cuda", 0),
        ("cuda:0", 0),
        ("cuda:2", 2),
        ("cuda2", 2),
    ],
)
def test_device_index(raw: str, expected: int | None) -> None:
    assert device_index(raw) == expected


def test_get_device_from_string() -> None:
    assert get_device("cpu") == "cpu"
    assert get_device("cuda") == "cuda:0"
    assert get_device("cuda1") == "cuda:1"


def test_get_device_from_object_with_device_attribute() -> None:
    obj = DummyDevice("cuda1")

    assert get_device(obj) == "cuda:1"


def test_get_device_rejects_object_without_device() -> None:
    with pytest.raises(TypeError):
        get_device(object())


def test_same_device_true() -> None:
    assert same_device("cuda", "cuda:0", DummyDevice("cuda0"))


def test_same_device_false() -> None:
    assert not same_device("cpu", "cuda")


def test_assert_same_device_passes() -> None:
    assert_same_device("cuda", "cuda:0", DummyDevice("cuda0"))


def test_assert_same_device_raises() -> None:
    with pytest.raises(DeviceMismatchError):
        assert_same_device("cpu", "cuda")