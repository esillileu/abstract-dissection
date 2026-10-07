"""Tests for publishing W2V1 analysis results to MLflow f2.w2v1.analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

from f2.suites.w2v1.publish_analysis import (
    ANALYSIS_EXPERIMENT_NAME,
    ensure_analysis_experiment,
    publish_table4_analysis,
    publish_table7_analysis,
)


def test_ensure_analysis_experiment_creates_or_returns() -> None:
    client = MagicMock()
    # Case 1: existing
    mock_exp = MagicMock(experiment_id="exp-123")
    client.get_experiment_by_name.return_value = mock_exp
    assert ensure_analysis_experiment(client) == "exp-123"

    # Case 2: new
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "exp-new"
    assert ensure_analysis_experiment(client) == "exp-new"
    client.create_experiment.assert_called_with(ANALYSIS_EXPERIMENT_NAME)


def test_publish_table7_analysis(tmp_path: Path) -> None:
    client = MagicMock()
    client.create_run.return_value = MagicMock(info=MagicMock(run_id="run-t7-1"))

    # Prepare mock summary.csv
    csv_file = tmp_path / "summary.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "architecture",
                "dimension",
                "training_words_millions",
                "epochs",
                "corpus",
                "seed",
                "mlflow_run_id",
                "overall_accuracy_percent",
                "dev_accuracy_percent",
                "test_accuracy_percent",
                "total_questions",
                "observed_training_seconds",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "architecture": "skipgram",
                "dimension": "640",
                "training_words_millions": "50",
                "epochs": "1",
                "corpus": "holmes",
                "seed": "1",
                "mlflow_run_id": "0123456789abcdef",
                "overall_accuracy_percent": "37.72",
                "dev_accuracy_percent": "38.50",
                "test_accuracy_percent": "36.94",
                "total_questions": "1040",
                "observed_training_seconds": "350.0",
            }
        )

    run_ids = publish_table7_analysis(client, "exp-analysis", tmp_path)
    assert len(run_ids) == 2  # 1 seed run + 1 summary run
    assert client.log_metric.called
    assert client.set_terminated.called


def test_publish_table4_analysis(tmp_path: Path) -> None:
    client = MagicMock()
    client.create_run.return_value = MagicMock(info=MagicMock(run_id="run-t4-1"))

    wmt_dir = tmp_path / "wmt" / "table4"
    wmt_dir.mkdir(parents=True)
    csv_file = wmt_dir / "runs.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "architecture",
                "dimension",
                "training_words_millions",
                "epochs",
                "corpus",
                "seed",
                "mlflow_run_id",
                "semantic_accuracy_percent",
                "syntactic_accuracy_percent",
                "total_accuracy_percent",
                "included_questions",
                "total_questions",
                "observed_training_seconds",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "architecture": "skipgram",
                "dimension": "300",
                "training_words_millions": "783",
                "epochs": "3",
                "corpus": "wmt",
                "seed": "1",
                "mlflow_run_id": "abcdef0123456789",
                "semantic_accuracy_percent": "54.0",
                "syntactic_accuracy_percent": "52.0",
                "total_accuracy_percent": "53.0",
                "included_questions": "19544",
                "total_questions": "19544",
                "observed_training_seconds": "15000.0",
            }
        )

    run_ids = publish_table4_analysis(client, "exp-analysis", tmp_path)
    assert len(run_ids) == 2  # 1 run + 1 summary run
    assert client.log_metric.called
