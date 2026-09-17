from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import Backend

from .base import TimeLayer
from .time_lstm import TimeLSTM


class TimeBiLSTM(TimeLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        stateful: bool = False,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(backend)
        self.forward_lstm = TimeLSTM(
            input_size,
            hidden_size,
            stateful=stateful,
            backend=self._backend,
        )
        self.backward_lstm = TimeLSTM(
            input_size,
            hidden_size,
            stateful=stateful,
            backend=self._backend,
        )

    def forward_manual(self, xs: Tensor) -> Tensor:
        out_forward = self.forward_lstm.forward(xs)
        out_backward = self.backward_lstm.forward(xs[:, ::-1])
        out_data = xs.backend.xp.concatenate(
            (out_forward.data, out_backward.data[:, ::-1]),
            axis=2,
        )
        return Tensor(out_data, backend=xs.backend)

    def backward_manual(self, dhs: Tensor) -> Tensor:
        hidden_size = dhs.shape[2] // 2
        dout_forward = dhs[:, :, :hidden_size]
        dout_backward = dhs[:, :, hidden_size:]

        dx_forward = self.forward_lstm.backward(dout_forward)
        dx_backward = self.backward_lstm.backward(dout_backward[:, ::-1])
        return Tensor(dx_forward.data + dx_backward.data[:, ::-1], backend=dhs.backend)

    def reset_state(self) -> None:
        self.forward_lstm.reset_state()
        self.backward_lstm.reset_state()

    def detach_state(self) -> None:
        self.forward_lstm.detach_state()
        self.backward_lstm.detach_state()
