from __future__ import annotations

from collections.abc import Sequence

from deepscratch.nn.layers import Affine, BatchNormalization, Dropout, Layer

from ..common import activation, initialize_affine
from ..sequential import Sequential


class MLP(Sequential):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        initializer: str = "he",
        activation_name: str = "relu",
        batchnorm: bool = False,
        dropout_ratio: float = 0.0,
        dropout_rng=None,
    ) -> None:
        layers: list[Layer] = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            affine = Affine(in_features, hidden_size)
            initialize_affine(affine, initializer)
            layers.append(affine)
            if batchnorm:
                layers.append(BatchNormalization())
            layers.append(activation(activation_name))
            if dropout_ratio > 0:
                layers.append(Dropout(dropout_ratio, rng=dropout_rng))
            in_features = hidden_size
        output = Affine(in_features, output_size)
        initialize_affine(output, initializer)
        layers.append(output)
        super().__init__(*layers)
