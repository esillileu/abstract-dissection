from ..core.tensor import Tensor
from .mnist import load_mnist as _load_mnist


def load_mnist(normalize=True, flatten=True, one_hot_label=False, gpu=False):
    return (
        (Tensor(j).gpu() if gpu else Tensor(j) for j in i)
        for i in _load_mnist(normalize, flatten, one_hot_label)
    )


__all__ = ["load_mnist"]
