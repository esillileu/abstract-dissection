from __future__ import annotations

import csv
from dataclasses import dataclass

import pytest

from f2.suites.w2v.observations import DenseObservationWriter, sparse_metric_rows


@dataclass
class Observation:
    epoch: int
    processed_tokens: int
    learning_rate: float = 0.05
    objective_loss_sum: float = 4.0
    objective_loss_count: int = 2
    elapsed_seconds: float = 1.0
    tokens_per_second: float = 10.0


def test_dense_observations_append_flush_and_sparse_points_match(tmp_path) -> None:
    path = tmp_path / "dense.csv"
    writer = DenseObservationWriter(path)
    first = writer.append([Observation(1, 10), Observation(1, 20)])
    second = writer.append([Observation(1, 30)])

    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    assert [int(record["processed_tokens"]) for record in records] == [10, 20, 30]
    assert set(records[0]) == {
        "epoch",
        "processed_tokens",
        "learning_rate",
        "objective_loss_sum",
        "objective_loss_count",
        "objective_loss",
        "elapsed_seconds",
        "tokens_per_second",
    }
    assert sparse_metric_rows((*first, *second), token_interval=15) == (
        (10, "train/objective_loss", 2.0),
        (10, "train/learning_rate", 0.05),
        (10, "runtime/tokens_per_second", 10.0),
        (30, "train/objective_loss", 2.0),
        (30, "train/learning_rate", 0.05),
        (30, "runtime/tokens_per_second", 10.0),
    )


def test_observation_schema_rejects_non_finite_values(tmp_path) -> None:
    writer = DenseObservationWriter(tmp_path / "dense.csv")
    with pytest.raises(ValueError, match="invalid Word2Vec observation"):
        writer.append([Observation(1, 10, objective_loss_sum=float("nan"))])
    assert not writer.path.exists()
