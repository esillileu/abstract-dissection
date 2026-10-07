"""Canonical publication of F2 W2V1 analysis results to MLflow experiment f2.w2v1.analysis."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from pathlib import Path

from mlflow import MlflowClient

from repro_core.context import RuntimePaths

ANALYSIS_EXPERIMENT_NAME = "f2.w2v1.analysis"


def ensure_analysis_experiment(client: MlflowClient) -> str:
    """Get or create the canonical f2.w2v1.analysis MLflow experiment."""
    exp = client.get_experiment_by_name(ANALYSIS_EXPERIMENT_NAME)
    if exp is not None:
        return exp.experiment_id
    return client.create_experiment(ANALYSIS_EXPERIMENT_NAME)


def publish_table7_analysis(
    client: MlflowClient,
    experiment_id: str,
    analysis_dir: Path,
) -> list[str]:
    """Publish Table 7 MSR Sentence Completion evaluation results to MLflow."""
    summary_csv = analysis_dir / "summary.csv"
    summary_md = analysis_dir / "summary.md"
    if not summary_csv.is_file():
        return []

    published_run_ids: list[str] = []
    with summary_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    now_ms = int(time.time() * 1000)

    # 1. Seed-level runs
    for row in rows:
        run_name = (
            f"table7-holmes-{row['architecture']}-d{row['dimension']}-s{row['seed']}"
        )
        tags = {
            "paper.id": "mikolov-2013-efficient-estimation",
            "paper.table": "7",
            "suite.name": "w2v1",
            "analysis.kind": "sentence_completion",
            "model.architecture": row["architecture"],
            "model.dimension": row["dimension"],
            "training.words_millions": row["training_words_millions"],
            "training.epochs": row["epochs"],
            "corpus": row["corpus"],
            "seed": row["seed"],
            "upstream.training_run_id": row["mlflow_run_id"],
        }
        run = client.create_run(
            experiment_id, start_time=now_ms, tags=tags, run_name=run_name
        )
        run_id = run.info.run_id

        client.log_metric(
            run_id,
            "eval/overall_accuracy_percent",
            float(row["overall_accuracy_percent"]),
        )
        client.log_metric(
            run_id, "eval/dev_accuracy_percent", float(row["dev_accuracy_percent"])
        )
        client.log_metric(
            run_id, "eval/test_accuracy_percent", float(row["test_accuracy_percent"])
        )
        client.log_metric(run_id, "eval/total_questions", int(row["total_questions"]))
        client.log_metric(
            run_id, "runtime/training_seconds", float(row["observed_training_seconds"])
        )
        client.set_terminated(run_id, status="FINISHED")
        published_run_ids.append(run_id)

    # 2. Table-level aggregate summary run
    if rows:
        scores = [float(r["overall_accuracy_percent"]) for r in rows]
        mean_acc = sum(scores) / len(scores)
        summary_run_name = "w2v1-table7-holmes-skipgram-d640-w50m-summary"
        summary_tags = {
            "paper.id": "mikolov-2013-efficient-estimation",
            "paper.table": "7",
            "suite.name": "w2v1",
            "analysis.kind": "summary",
            "model.architecture": "skipgram",
            "model.dimension": "640",
            "corpus": "holmes",
            "seed_count": str(len(rows)),
        }
        s_run = client.create_run(
            experiment_id,
            start_time=now_ms,
            tags=summary_tags,
            run_name=summary_run_name,
        )
        s_id = s_run.info.run_id
        client.log_metric(s_id, "eval/overall_accuracy_mean", round(mean_acc, 2))
        client.log_metric(s_id, "paper/reported_accuracy", 48.0)
        if summary_md.is_file():
            client.log_artifact(s_id, str(summary_md))
        client.log_artifact(s_id, str(summary_csv))
        client.set_terminated(s_id, status="FINISHED")
        published_run_ids.append(s_id)

    return published_run_ids


def _publish_analogy_table(
    client: MlflowClient,
    experiment_id: str,
    base_dir: Path,
    *,
    table_number: str,
    get_table_dir: Callable[[Path, str], Path],
    csv_filename: str,
    summary_filename: str,
    extract_identity: Callable[[str, dict[str, str]], tuple[str, dict[str, str]]],
    corpora: tuple[str, ...] = ("wmt", "lm1b", "umbc"),
) -> list[str]:
    """Publish a Word Analogy evaluation table with common MLflow run lifecycle."""
    published: list[str] = []
    now_ms = int(time.time() * 1000)

    for corpus in corpora:
        c_dir = get_table_dir(base_dir, corpus)
        runs_csv = c_dir / csv_filename
        summary_md = c_dir / summary_filename
        if not runs_csv.is_file():
            continue

        with runs_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            run_name, extra_tags = extract_identity(corpus, row)
            tags = {
                "paper.id": "mikolov-2013-efficient-estimation",
                "paper.table": table_number,
                "suite.name": "w2v1",
                "analysis.kind": "word_analogy",
                **extra_tags,
                "corpus": corpus,
                "seed": row["seed"],
                "upstream.training_run_id": row["mlflow_run_id"],
            }
            run = client.create_run(
                experiment_id, start_time=now_ms, tags=tags, run_name=run_name
            )
            run_id = run.info.run_id

            client.log_metric(
                run_id,
                "eval/total_accuracy_percent",
                float(row["total_accuracy_percent"]),
            )
            client.log_metric(
                run_id,
                "eval/semantic_accuracy_percent",
                float(row["semantic_accuracy_percent"]),
            )
            client.log_metric(
                run_id,
                "eval/syntactic_accuracy_percent",
                float(row["syntactic_accuracy_percent"]),
            )
            client.log_metric(
                run_id, "eval/included_questions", int(row["included_questions"])
            )
            client.log_metric(
                run_id,
                "runtime/training_seconds",
                float(row["observed_training_seconds"]),
            )
            client.set_terminated(run_id, status="FINISHED")
            published.append(run_id)

        # Summary run
        s_run = client.create_run(
            experiment_id,
            start_time=now_ms,
            tags={"paper.table": table_number, "corpus": corpus, "suite.name": "w2v1"},
            run_name=f"w2v1-table{table_number}-{corpus}-summary",
        )
        s_id = s_run.info.run_id
        if summary_md.is_file():
            client.log_artifact(s_id, str(summary_md))
        client.log_artifact(s_id, str(runs_csv))
        client.set_terminated(s_id, status="FINISHED")
        published.append(s_id)

    return published


def _table2_identity(corpus: str, row: dict[str, str]) -> tuple[str, dict[str, str]]:
    dim = row["vector_dimension"]
    words = row["training_words_millions"]
    return (
        f"table2-{corpus}-cbow-d{dim}-w{words}m-s{row['seed']}",
        {
            "model.architecture": "cbow",
            "model.dimension": dim,
            "training.words_millions": words,
        },
    )


def _table3_identity(corpus: str, row: dict[str, str]) -> tuple[str, dict[str, str]]:
    arch = row.get("architecture") or row.get("model_kind", "unknown")
    dim = row.get("dimension") or row.get("vector_dimension", "unknown")
    return (
        f"table3-{corpus}-{arch}-d{dim}-s{row['seed']}",
        {
            "model.architecture": arch,
            "model.dimension": dim,
            "training.words_millions": row["training_words_millions"],
        },
    )


def _table4_identity(corpus: str, row: dict[str, str]) -> tuple[str, dict[str, str]]:
    dim = row.get("dimension", "300")
    words = row.get("training_words_millions", "783")
    epochs = row.get("epochs", "3")
    return (
        f"table4-{corpus}-{row['architecture']}-d{dim}-s{row['seed']}",
        {
            "model.architecture": row["architecture"],
            "model.dimension": str(dim),
            "training.words_millions": str(words),
            "training.epochs": str(epochs),
        },
    )


def _table5_identity(corpus: str, row: dict[str, str]) -> tuple[str, dict[str, str]]:
    return (
        f"table5-{corpus}-{row['architecture']}-d{row['dimension']}-w{row['training_words_millions']}m-s{row['seed']}",
        {
            "model.architecture": row["architecture"],
            "model.dimension": row["dimension"],
            "training.words_millions": row["training_words_millions"],
            "training.epochs": row["epochs"],
        },
    )


def publish_table2_analysis(
    client: MlflowClient,
    experiment_id: str,
    base_dir: Path,
) -> list[str]:
    """Publish Table 2 scaling results to MLflow."""
    return _publish_analogy_table(
        client,
        experiment_id,
        base_dir,
        table_number="2",
        get_table_dir=lambda base, corpus: base / corpus,
        csv_filename="table2-runs.csv",
        summary_filename="summary.md",
        extract_identity=_table2_identity,
    )


def publish_table3_analysis(
    client: MlflowClient,
    experiment_id: str,
    base_dir: Path,
) -> list[str]:
    """Publish Table 3 vector dimension / architecture comparison results to MLflow."""
    return _publish_analogy_table(
        client,
        experiment_id,
        base_dir,
        table_number="3",
        get_table_dir=lambda base, corpus: base / corpus,
        csv_filename="table3-runs.csv",
        summary_filename="table3-summary.md",
        extract_identity=_table3_identity,
    )


def publish_table4_analysis(
    client: MlflowClient,
    experiment_id: str,
    base_dir: Path,
) -> list[str]:
    """Publish Table 4 Word Analogy (Words vs Epochs) results to MLflow."""
    return _publish_analogy_table(
        client,
        experiment_id,
        base_dir,
        table_number="4",
        get_table_dir=lambda base, corpus: base / corpus / "table4",
        csv_filename="runs.csv",
        summary_filename="summary.md",
        extract_identity=_table4_identity,
    )


def publish_table5_analysis(
    client: MlflowClient,
    experiment_id: str,
    base_dir: Path,
) -> list[str]:
    """Publish Table 5 architecture comparison results to MLflow."""
    return _publish_analogy_table(
        client,
        experiment_id,
        base_dir,
        table_number="5",
        get_table_dir=lambda base, corpus: base / corpus / "table5",
        csv_filename="runs.csv",
        summary_filename="summary.md",
        extract_identity=_table5_identity,
    )


def publish_all_w2v1_analysis(
    tracking_uri: str,
    paths: RuntimePaths | None = None,
) -> dict[str, list[str]]:
    """Publish all available W2V1 analysis tables into MLflow f2.w2v1.analysis."""
    paths = paths or RuntimePaths.from_environment()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = ensure_analysis_experiment(client)

    w2v1_analysis_dir = paths.analysis_output("f2", "w2v1")
    t7_analysis_dir = paths.analysis_output("f2", "table7")

    results: dict[str, list[str]] = {}

    results["table2"] = publish_table2_analysis(
        client, experiment_id, w2v1_analysis_dir
    )
    results["table3"] = publish_table3_analysis(
        client, experiment_id, w2v1_analysis_dir
    )
    results["table4"] = publish_table4_analysis(
        client, experiment_id, w2v1_analysis_dir
    )
    results["table5"] = publish_table5_analysis(
        client, experiment_id, w2v1_analysis_dir
    )
    results["table7"] = publish_table7_analysis(client, experiment_id, t7_analysis_dir)

    return results


__all__ = [
    "ANALYSIS_EXPERIMENT_NAME",
    "ensure_analysis_experiment",
    "publish_all_w2v1_analysis",
    "publish_table2_analysis",
    "publish_table3_analysis",
    "publish_table4_analysis",
    "publish_table5_analysis",
    "publish_table7_analysis",
]
