"""Hierarchical traversal and attribute resolution for layers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ...types.parameter import Parameter

if TYPE_CHECKING:
    from .layer import Layer


def _layer_cls() -> type[Layer]:
    from .layer import Layer

    return Layer


def _iter_named_parameters(
    value: Any,
    prefix: str,
    seen: set[int],
) -> Iterator[tuple[str, Parameter]]:
    if isinstance(value, Parameter):
        if not value.requires_grad:
            return

        param_id = id(value)

        if param_id in seen:
            return

        seen.add(param_id)
        yield prefix, value
        return

    layer_type = _layer_cls()
    if isinstance(value, layer_type):
        for name, item in value.__dict__.items():
            child_prefix = f"{prefix}.{name}"
            yield from _iter_named_parameters(
                value=item,
                prefix=child_prefix,
                seen=seen,
            )

        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            item_prefix = f"{prefix}.{index}"
            yield from _iter_named_parameters(
                value=item,
                prefix=item_prefix,
                seen=seen,
            )

        return

    if isinstance(value, dict):
        for key, item in value.items():
            item_prefix = f"{prefix}.{key}"
            yield from _iter_named_parameters(
                value=item,
                prefix=item_prefix,
                seen=seen,
            )


def _iter_layers(value: Any, seen: set[int]) -> Iterator[Layer]:
    layer_type = _layer_cls()
    if isinstance(value, layer_type):
        layer_id = id(value)

        if layer_id in seen:
            return

        seen.add(layer_id)
        yield value
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_layers(item, seen)

        return

    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_layers(item, seen)


def _iter_named_buffers(
    layer: Layer,
    prefix: str,
    seen_layers: set[int],
    runtime_state: bool | None,
) -> Iterator[tuple[str, Any]]:
    layer_id = id(layer)
    if layer_id in seen_layers:
        return
    seen_layers.add(layer_id)

    for name, is_runtime in getattr(layer, "_buffers", {}).items():
        if runtime_state is None or runtime_state == is_runtime:
            yield f"{prefix}{name}", getattr(layer, name)

    layer_type = _layer_cls()
    for name, value in layer.__dict__.items():
        child_prefix = f"{prefix}{name}."
        if isinstance(value, layer_type):
            yield from _iter_named_buffers(
                value, child_prefix, seen_layers, runtime_state
            )
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if isinstance(child, layer_type):
                    yield from _iter_named_buffers(
                        child,
                        f"{child_prefix}{index}.",
                        seen_layers,
                        runtime_state,
                    )
        elif isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, layer_type):
                    yield from _iter_named_buffers(
                        child,
                        f"{child_prefix}{key}.",
                        seen_layers,
                        runtime_state,
                    )


def _resolve_owner(root: Layer, path: str) -> tuple[Layer, str]:
    layer_type = _layer_cls()
    parts = path.split(".")
    value: Any = root
    for part in parts[:-1]:
        if isinstance(value, (list, tuple)):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            value = getattr(value, part)
    if not isinstance(value, layer_type):
        raise TypeError(f"buffer owner for {path!r} is not a Layer")
    return value, parts[-1]
