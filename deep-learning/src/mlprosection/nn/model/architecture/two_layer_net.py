"""The book's two-affine-layer MNIST network."""

from __future__ import annotations

from mlprosection.nn.layers import Affine, Relu
from mlprosection.nn.model.sequential import Sequential

from ..common import initialize_affine


class TwoLayerNet(Sequential):
    """Affine(784, 50) -> ReLU -> Affine(50, 10).

    ``numerical_gradient`` is deliberately an explicit model capability.  It
    is used by the DS1 numerical-gradient experiment; ordinary training keeps
    using the layer backward implementations.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        output_size: int,
        initializer: str = "std:0.01",
        numerical_step: float = 1e-4,
    ) -> None:
        first = Affine(input_size, hidden_size)
        second = Affine(hidden_size, output_size)
        initialize_affine(first, initializer)
        initialize_affine(second, initializer)
        super().__init__(first, Relu(), second)
        self.numerical_step = numerical_step

    def numerical_gradient(self, x, target, objective) -> None:
        """Write central-difference gradients into all model parameters."""
        h = self.numerical_step
        for _name, parameter in self.named_parameters():
            values = parameter.data
            gradient = parameter.backend.xp.zeros_like(values)
            iterator = parameter.backend.xp.nditer(
                values, flags=["multi_index"], op_flags=["readwrite"]
            )
            while not iterator.finished:
                index = iterator.multi_index
                original = values[index]
                values[index] = original + h
                plus = objective.forward(self.forward(x, cache=False), target, cache=False).loss
                values[index] = original - h
                minus = objective.forward(self.forward(x, cache=False), target, cache=False).loss
                values[index] = original
                gradient[index] = (plus.data - minus.data) / (2 * h)
                iterator.iternext()
            parameter.grad[...] = gradient
