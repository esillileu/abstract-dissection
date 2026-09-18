"""Durable staging schema for dense Word2Vec training observations."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


class EngineObservation(Protocol):
    epoch: int
    processed_tokens: int
    learning_rate: float
    objective_loss_sum: float
    objective_loss_count: int
    elapsed_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class ObservationRow:
    epoch: int
    processed_tokens: int
    learning_rate: float
    objective_loss_sum: float
    objective_loss_count: int
    elapsed_seconds: float
    tokens_per_second: float

    @classmethod
    def from_engine(cls, observation: EngineObservation) -> ObservationRow:
        row = cls(
            epoch=int(observation.epoch),
            processed_tokens=int(observation.processed_tokens),
            learning_rate=float(observation.learning_rate),
            objective_loss_sum=float(observation.objective_loss_sum),
            objective_loss_count=int(observation.objective_loss_count),
            elapsed_seconds=float(observation.elapsed_seconds),
            tokens_per_second=float(observation.tokens_per_second),
        )
        row.validate()
        return row

    @property
    def objective_loss(self) -> float:
        return self.objective_loss_sum / self.objective_loss_count

    def validate(self) -> None:
        if (
            self.epoch < 1
            or self.processed_tokens < 0
            or self.objective_loss_count < 1
            or self.learning_rate <= 0.0
            or self.elapsed_seconds < 0.0
            or self.tokens_per_second < 0.0
            or not all(
                math.isfinite(value)
                for value in (
                    self.learning_rate,
                    self.objective_loss_sum,
                    self.elapsed_seconds,
                    self.tokens_per_second,
                    self.objective_loss,
                )
            )
        ):
            raise ValueError("invalid Word2Vec observation")


_COLUMNS = (
    "epoch",
    "processed_tokens",
    "learning_rate",
    "objective_loss_sum",
    "objective_loss_count",
    "objective_loss",
    "elapsed_seconds",
    "tokens_per_second",
)


class DenseObservationWriter:
    """Append observations to ephemeral staging and flush each accepted batch."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(
        self, observations: Iterable[EngineObservation]
    ) -> tuple[ObservationRow, ...]:
        rows = tuple(ObservationRow.from_engine(item) for item in observations)
        if not rows:
            return rows
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if new_file:
                writer.writerow(_COLUMNS)
            for row in rows:
                writer.writerow(
                    (
                        row.epoch,
                        row.processed_tokens,
                        row.learning_rate,
                        row.objective_loss_sum,
                        row.objective_loss_count,
                        row.objective_loss,
                        row.elapsed_seconds,
                        row.tokens_per_second,
                    )
                )
            stream.flush()
            os.fsync(stream.fileno())
        return rows


def sparse_metric_rows(
    observations: Iterable[ObservationRow], *, token_interval: int
) -> tuple[tuple[int, str, float], ...]:
    """Project dense points to MLflow rows without inventing new step values."""
    if token_interval < 1:
        raise ValueError("token_interval must be at least 1")
    selected: list[ObservationRow] = []
    next_token = 0
    for row in observations:
        row.validate()
        if row.processed_tokens >= next_token:
            selected.append(row)
            next_token = row.processed_tokens + token_interval
    return tuple(
        metric
        for row in selected
        for metric in (
            (row.processed_tokens, "train/objective_loss", row.objective_loss),
            (row.processed_tokens, "train/learning_rate", row.learning_rate),
            (row.processed_tokens, "runtime/tokens_per_second", row.tokens_per_second),
        )
    )
