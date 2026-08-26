from __future__ import annotations

from typing import TYPE_CHECKING

from .parameter import Parameter

if TYPE_CHECKING:
    from ..layers.activation import Activation
    from ..layers.base import Layer, NamedParameter, NamedParameters

    __all__ = [
        "Activation",
        "Layer",
        "NamedParameter",
        "NamedParameters",
        "Parameter",
    ]

__all__ = ["Parameter"]
