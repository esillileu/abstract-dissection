from __future__ import annotations

from mlprosection.experiment.reproducibility import configure_runtime, seed_streams
from mlprosection.nn.model import MLP


def test_model_initialization_is_reproducible_before_model_creation() -> None:
    config = {"seed": 1208965604, "numerics": {"device": "cpu", "dtype": "float32", "backend": "numpy"}}
    configure_runtime(config)
    first = MLP(input_size=4, hidden_sizes=[3], output_size=2)
    first_values = [parameter.backend.to_numpy(parameter.data).copy() for _, parameter in first.named_parameters()]

    configure_runtime(config)
    second = MLP(input_size=4, hidden_sizes=[3], output_size=2)
    second_values = [parameter.backend.to_numpy(parameter.data).copy() for _, parameter in second.named_parameters()]

    assert all((left == right).all() for left, right in zip(first_values, second_values, strict=True))


def test_named_seed_streams_are_stable_and_distinct() -> None:
    first = seed_streams(1208965604)
    second = seed_streams(1208965604)

    assert first == second
    assert len(set(first.__dict__.values())) == len(first.__dict__)
