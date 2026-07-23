from __future__ import annotations

from typing import TYPE_CHECKING

from .parameter import Parameter

if TYPE_CHECKING:
    from ..layers.base import Layer, NamedParameter, NamedParameters
    from ..layers.activation import Activation

    __all__ = [
        "Parameter",
        "Layer",
        "Activation",
        "NamedParameter",
        "NamedParameters",
    ]

__all__ = ["Parameter"]
