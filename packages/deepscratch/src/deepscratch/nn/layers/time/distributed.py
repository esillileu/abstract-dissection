from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend, get_default_backend, resolve_backend

from ...types import Parameter
from ..base import Layer
from ..embeding import Embedding
from ..linear import Affine
from ..regulizer import BatchNormalization, Dropout
from .base import TimeLayer


class TimeDistributed(TimeLayer):
    """Apply one ordinary layer to every timestep with shared parameters.

    The wrapped layer is called once on a flattened ``batch * time`` batch, so
    its parameter gradients naturally aggregate across all timesteps.
    """

    def __init__(self, layer: Layer) -> None:
        super().__init__(layer.backend)
        self.layer = layer
        self.input_shape: tuple[int, ...] | None = None
        self.output_shape: tuple[int, ...] | None = None

    def forward_manual(self, xs: Tensor, *, cache: bool = True) -> Tensor:
        if xs.ndim < 2:
            raise ValueError("TimeDistributed expects (batch, time, ...) input")
        self.input_shape = xs.shape
        flat = Tensor(
            xs.data.reshape(xs.shape[0] * xs.shape[1], *xs.shape[2:]),
            backend=xs.backend,
        )
        output = self.layer.forward(flat)
        self.output_shape = output.shape[1:]
        result = Tensor(
            output.data.reshape(xs.shape[0], xs.shape[1], *self.output_shape),
            backend=output.backend,
        )
        if not cache:
            self.input_shape = None
            self.output_shape = None
            for name in ("x", "idx", "mask", "cache"):
                if hasattr(self.layer, name):
                    setattr(self.layer, name, None)
        return result

    def backward_manual(self, dout: Tensor):
        if self.input_shape is None or self.output_shape is None:
            raise RuntimeError("forward must be called before backward")
        flat = Tensor(
            dout.data.reshape(
                self.input_shape[0] * self.input_shape[1], *self.output_shape
            ),
            backend=dout.backend,
        )
        dx = self.layer.backward(flat)
        if dx is None:
            return None
        return Tensor(dx.data.reshape(*self.input_shape), backend=dx.backend)


class TimeEmbedding(TimeDistributed):
    def __init__(
        self,
        vocab_size: int,
        wordvec_size: int,
        *,
        backend: Backend | str | None = None,
        weight_scale: float = 0.01,
    ) -> None:
        resolved = (
            resolve_backend(backend) if backend is not None else get_default_backend()
        )
        layer = Embedding(vocab_size, wordvec_size, backend=resolved)
        super().__init__(layer)
        self.weight_scale = weight_scale
        if weight_scale != 0.01:
            self.W.data[...] *= weight_scale / 0.01

    @property
    def W(self) -> Parameter:
        return self.layer.W


class TimeAffine(TimeDistributed):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        backend: Backend | str | None = None,
        weight_scale: float | None = None,
        weight: Parameter | None = None,
        transpose_weight: bool = False,
    ) -> None:
        resolved = backend or (weight.backend if weight is not None else None)
        layer = Affine(
            in_features,
            out_features,
            backend=resolved,
            weight=weight,
            transpose_weight=transpose_weight,
        )
        super().__init__(layer)
        if weight_scale is not None and weight is None:
            self.W.data[...] *= weight_scale / 0.01

    @property
    def W(self) -> Parameter:
        return self.layer.W

    @property
    def b(self) -> Parameter:
        return self.layer.b


class TimeDropout(TimeDistributed):
    def __init__(self, dropout_ratio: float = 0.5, *, rng=None) -> None:
        super().__init__(Dropout(dropout_ratio, inverted=True, rng=rng))
        self.dropout_ratio = dropout_ratio


class TimeBatchNormalization(TimeDistributed):
    """Batch-normalize over the flattened batch-and-time population."""

    def __init__(self, **kwargs) -> None:
        super().__init__(BatchNormalization(**kwargs))


SimpleTimeAffine = TimeAffine
