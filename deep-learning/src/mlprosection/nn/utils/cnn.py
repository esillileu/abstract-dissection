from typing import Any, Tuple, TypeAlias

IntPair: TypeAlias = Tuple[int, int]


def im2col(
    x: Any,
    kernel_size: IntPair,
    stride: IntPair = 1,
    padding: IntPair = 0,
    *,
    xp: Any,
    pad_value: int | float = 0,
) -> Any:
    kernel_h, kernel_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)

    batch_size, channels, input_h, input_w = x.shape

    output_h = _calculate_output_size(
        input_size=input_h,
        kernel_size=kernel_h,
        stride=stride_h,
        padding=pad_h,
    )
    output_w = _calculate_output_size(
        input_size=input_w,
        kernel_size=kernel_w,
        stride=stride_w,
        padding=pad_w,
    )

    padded = xp.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (pad_h, pad_h),
            (pad_w, pad_w)
        ),
        mode="constant",
        constant_values=pad_value,
    )

    col = xp.empty(
        (
            batch_size,
            channels,
            kernel_h,
            kernel_w,
            output_h,
            output_w,
        ),
        dtype=x.dtype,
    )

    for kernel_y in range(kernel_h):
        y_end = kernel_y + stride_h * output_h

        for kernel_x in range(kernel_w):
            x_end = kernel_x + stride_w * output_w

            col[
                :,
                :,
                kernel_y,
                kernel_x,
                :,
                :,
            ] = padded[
                :,
                :,
                kernel_y:y_end:stride_h,
                kernel_x:x_end:stride_w,
            ]

    col = xp.transpose(
        col,
        (0, 4, 5, 1, 2, 3),
    )

    return col.reshape(
        batch_size * output_h * output_w,
        channels * kernel_h * kernel_w,
    )


def col2im(
    col: Any,
    input_shape: tuple[int, int, int, int],
    kernel_size: IntPair,
    stride: IntPair = 1,
    padding: IntPair = 0,
    *,
    xp: Any,
) -> Any:
    kernel_h, kernel_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)

    batch_size, channels, input_h, input_w = input_shape

    output_h = _calculate_output_size(
        input_size=input_h,
        kernel_size=kernel_h,
        stride=stride_h,
        padding=pad_h,
    )
    output_w = _calculate_output_size(
        input_size=input_w,
        kernel_size=kernel_w,
        stride=stride_w,
        padding=pad_w,
    )

    col = col.reshape(
        batch_size,
        output_h,
        output_w,
        channels,
        kernel_h,
        kernel_w,
    )
    col = xp.transpose(
        col,
        (0, 3, 4, 5, 1, 2),
    )

    padded_h = input_h + 2 * pad_h
    padded_w = input_w + 2 * pad_w

    image = xp.zeros(
        (
            batch_size,
            channels,
            padded_h,
            padded_w,
        ),
        dtype=col.dtype,
    )

    for kernel_y in range(kernel_h):
        y_end = kernel_y + stride_h * output_h

        for kernel_x in range(kernel_w):
            x_end = kernel_x + stride_w * output_w

            image[
                :,
                :,
                kernel_y:y_end:stride_h,
                kernel_x:x_end:stride_w,
            ] += col[
                :,
                :,
                kernel_y,
                kernel_x,
                :,
                :,
            ]

    return image[
        :,
        :,
        pad_h : pad_h + input_h,
        pad_w : pad_w + input_w,
    ]


def _pair(value: IntPair) -> tuple[int, int]:
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"value must be non-negative, got {value}")

        return value, value

    if len(value) != 2:
        raise ValueError(f"expected int or tuple of length 2, got {value}")

    first, second = value

    if first < 0 or second < 0:
        raise ValueError(f"values must be non-negative, got {value}")

    return first, second


def _calculate_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> int:
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")

    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    numerator = input_size + 2 * padding - kernel_size

    if numerator < 0:
        raise ValueError(
            "kernel size cannot be larger than the padded input: "
            f"input={input_size}, kernel={kernel_size}, "
            f"padding={padding}"
        )

    return numerator // stride + 1
