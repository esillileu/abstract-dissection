from __future__ import annotations

from typing import TYPE_CHECKING

from .parameter import Parameter

if TYPE_CHECKING:
    from ..layers.base import Layer, NamedParameter, NamedParameters
    from ..layers.activation import Activation
    from ..layers.criterion import Criterion

    __all__ = [
        "Parameter",
        "Layer",
        "Activation",
        "Criterion",
        "NamedParameter",
        "NamedParameters",
    ]

__all__ = ["Parameter"]
