from typing import Protocol

from mlprosection import Tensor


class Initializer(Protocol):
    def __call__(self, tensor: Tensor, /) -> None: ...


def xavier_normal_(tensor: Tensor) -> Tensor:
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = (2.0 / (fan_in + fan_out)) ** 0.5

    xp = tensor.backend.xp
    values = xp.random.randn(*tensor.shape) * std

    tensor.data[...] = values.astype(tensor.dtype, copy=False)

    return tensor


def he_normal_(tensor: Tensor) -> Tensor:
    fan_in, _ = calculate_fan_in_and_fan_out(tensor)
    std = (2.0 / fan_in) ** 0.5

    xp = tensor.backend.xp
    values = xp.random.randn(*tensor.shape) * std

    tensor.data[...] = values.astype(tensor.dtype, copy=False)

    return tensor


def calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    if tensor.ndim < 2:
        raise ValueError("Fan in and fan out require at least 2 dimensions.")

    receptive_field_size = 1

    for size in tensor.shape[2:]:
        receptive_field_size *= size

    if tensor.ndim == 2:
        fan_in, fan_out = tensor.shape
    else:
        fan_in = tensor.shape[1] * receptive_field_size
        fan_out = tensor.shape[0] * receptive_field_size

    return fan_in, fan_out
