import numpy as np

from mlprosection.nn.types import Parameter
from mlprosection.profiling.utils import count_optimizer_state_bytes


class OptimizerWithState:
    def __init__(self, parameter: Parameter) -> None:
        self.params = [("W", parameter)]
        self.state = {"W": np.zeros_like(parameter.data)}


class OptimizerWithoutState:
    def __init__(self, parameter: Parameter) -> None:
        self.params = [("W", parameter)]


def test_count_optimizer_state_bytes_excludes_params_and_grads() -> None:
    parameter = Parameter(np.ones((2, 3)))
    optimizer = OptimizerWithoutState(parameter)

    assert count_optimizer_state_bytes(optimizer) == 0


def test_count_optimizer_state_bytes_counts_state_arrays() -> None:
    parameter = Parameter(np.ones((2, 3)))
    optimizer = OptimizerWithState(parameter)

    assert count_optimizer_state_bytes(optimizer) == parameter.data.nbytes
