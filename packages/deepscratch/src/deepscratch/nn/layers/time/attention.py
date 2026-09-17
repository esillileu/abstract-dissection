from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend

from .base import TimeLayer


class TimeAttention(TimeLayer):
    """Dot-product attention over encoder and decoder time states."""

    def __init__(self, *, backend: Backend | str | None = None) -> None:
        super().__init__(backend)
        self.cache: tuple[Tensor, Tensor] | None = None
        self.weights = None

    def forward_manual(
        self,
        enc_hs: Tensor,
        dec_hs: Tensor,
        *,
        cache: bool = True,
    ) -> Tensor:
        xp = enc_hs.backend.xp
        scores = xp.sum(enc_hs.data[:, None, :, :] * dec_hs.data[:, :, None, :], axis=3)
        scores -= scores.max(axis=2, keepdims=True)
        weights = xp.exp(scores)
        weights /= weights.sum(axis=2, keepdims=True)
        self.weights = weights if cache else None
        self.cache = (enc_hs, dec_hs) if cache else None
        context = xp.sum(weights[:, :, :, None] * enc_hs.data[:, None, :, :], axis=2)
        return Tensor(context, backend=enc_hs.backend)

    def backward_manual(self, dout: Tensor) -> tuple[Tensor, Tensor]:
        if self.cache is None or self.weights is None:
            raise RuntimeError("forward must be called before backward")
        enc_hs, dec_hs = self.cache
        xp = dout.backend.xp
        dweights = xp.sum(dout.data[:, :, None, :] * enc_hs.data[:, None, :, :], axis=3)
        denc = xp.sum(self.weights[:, :, :, None] * dout.data[:, :, None, :], axis=1)
        dscores = self.weights * (
            dweights - xp.sum(dweights * self.weights, axis=2, keepdims=True)
        )
        denc += xp.sum(dscores[:, :, :, None] * dec_hs.data[:, None, :, :], axis=1)
        ddec = xp.sum(dscores[:, :, :, None] * enc_hs.data[:, None, :, :], axis=2)
        return Tensor(denc, backend=dout.backend), Tensor(ddec, backend=dout.backend)
