from .types import DeviceMismatchError, DeviceTarget


def normalize_device(device: str) -> str:
    value = device.lower().replace(" ", "")

    if value == "cpu":
        return "cpu"

    if value in {"gpu", "cuda"}:
        return "cuda:0"

    if value.startswith("cuda:"):
        suffix = value.split(":", maxsplit=1)[1]
        if not suffix.isdigit():
            raise ValueError(f"invalid cuda device: {device}")
        return f"cuda:{int(suffix)}"

    if value.startswith("cuda") and value[4:].isdigit():
        return f"cuda:{int(value[4:])}"

    raise ValueError(
        "device must be one of: 'cpu', 'gpu', 'cuda', 'cuda:0', 'cuda0', ..."
    )


def device_index(device: str) -> int | None:
    normalized = normalize_device(device)

    if normalized == "cpu":
        return None

    return int(normalized.split(":", maxsplit=1)[1])


def get_device(target: DeviceTarget) -> str:
    if isinstance(target, str):
        return normalize_device(target)

    if hasattr(target, "device"):
        return normalize_device(target.device)

    raise TypeError(f"object has no device: {type(target)!r}")


def same_device(*targets: DeviceTarget) -> bool:
    if len(targets) <= 1:
        return True

    first = get_device(targets[0])

    for target in targets[1:]:
        if get_device(target) != first:
            return False

    return True


def assert_same_device(*targets: DeviceTarget) -> None:
    if len(targets) <= 1:
        return

    devices = [get_device(target) for target in targets]
    first = devices[0]

    for device in devices[1:]:
        if device != first:
            raise DeviceMismatchError(
                f"device mismatch: expected {first}, got {device}"
            )