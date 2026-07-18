from .core.tensor import Tensor


def exp(x: Tensor) -> Tensor:
    return x.exp()


def log(x: Tensor) -> Tensor:
    return x.log()


def sqrt(x: Tensor) -> Tensor:
    return x.sqrt()


def relu(x: Tensor):
    return x[x <= 0]


def sigmoid(x: Tensor):
    return 1 / (1 + (-x).exp())


def softmax(x: Tensor):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = x.exp()
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - x.max()
        x = x.exp() / x.exp().sum()

    return x


def cee(y: Tensor, t: Tensor):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return (
        -(y[Tensor.arange(batch_size, backend=y.backend), t] + 1e-7).log().sum()
        / batch_size
    )
