from __future__ import annotations

from typing import Any

from mlprosection.core import Tensor, Backend


class Parameter(Tensor):
    def __init__(self, data: Any, backend: Backend|None = None, requires_grad:bool = True, name: str | None = None) -> None:
        super().__init__(data, backend=backend, requires_grad=requires_grad, name=name)
        self.grad = self.backend.xp.zeros_like(self.data)
