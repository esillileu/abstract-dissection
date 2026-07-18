from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tqdm import tqdm

from .base import Trainer

if TYPE_CHECKING:
    from mlprosection import Tensor
    from mlprosection.nn.types import Layer, Criterion
    from mlprosection.optim import Optimizer


class ForwardTrainer(Trainer):
    def __init__(
        self,
        model: Layer,
        criterion: Criterion,
        optimizer: Optimizer,
        max_epoch: int = 10,
        batch_size: int = 32,
        log_interval: int = 20,
        max_grad: float | None = None,
        drop_last: bool | None = False,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            max_epoch=max_epoch,
            batch_size=batch_size,
            log_interval=log_interval,
            drop_last=drop_last,
        )
        self.max_grad = max_grad

        self.pbar: tqdm | None = None
        self.epoch: int | None = None

    def fit(
        self,
        x_train: Tensor,
        t_train: Tensor,
        x_val: Tensor | None = None,
        t_val: Tensor | None = None,
    ) -> None:
        xp = self.backend.xp
        self.start_time = time.time()
        self.train = True
        skip_validation = x_val is None or t_val is None

        data_size = len(x_train)
        max_iters = self.num_batches(data_size)
        total_steps = self.max_epoch * max_iters

        self.pbar = tqdm(total=total_steps, desc="train", unit="step")
        with self.pbar:
            for epoch in range(self.max_epoch):
                self.epoch = epoch + 1
                idx = xp.random.permutation(xp.arange(data_size))
                shuffled_x, shuffled_t = x_train[idx], t_train[idx]
                self.run_epoch(shuffled_x, shuffled_t)

                if not skip_validation:
                    self.train = False
                    self.run_epoch(x_val, t_val)
                    self.train = True
