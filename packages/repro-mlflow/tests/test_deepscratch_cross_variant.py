import csv
from pathlib import Path

from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from dlfs.analysis.orchestrator import write_analysis
from dlfs.identity import Variant, Volume
from repro_core.cli import app


def test_comparison_table_marks_absent_native_metric_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_F1_URL", uri)
    client = MlflowClient(uri)
    canonical_id = client.create_experiment("deepscratch.ds2")
    common = {
        "run.type": "seed_trial",
        "experiment.id": "e03",
        "master_seed": "1",
        "protocol.version": "book-source-v1",
        "runtime.device": "cuda:0",
    }
    implemented = client.create_run(
        canonical_id,
        tags={
            **common,
            "condition.id": "LM-SMALL-RNN",
            "atomic_run.id": "LM-SMALL-RNN",
            "implementation.variant": "implemented",
            "result.schema.name": "ds2-implemented",
            "result.schema.version": "1",
            "result.durable_complete": "true",
        },
    )
    client.log_metric(implemented.info.run_id, "final/train/perplexity", 120.0)
    client.set_terminated(implemented.info.run_id)
    original = client.create_run(
        canonical_id,
        tags={
            **common,
            "condition.id": "SMALL-RNNLM",
            "atomic_run.id": "SMALL-RNNLM",
            "implementation.variant": "original",
            "result.schema.name": "ds2-original",
            "result.schema.version": "1",
        },
    )
    client.set_terminated(original.info.run_id)

    output = write_analysis(
        uri,
        volume=Volume.DS2,
        experiment_ids=["e03"],
        variants=(Variant.IMPLEMENTED, Variant.ORIGINAL),
        output_dir=tmp_path / "comparison",
        cache_dir=tmp_path / "cache",
    )

    assert output == tmp_path / "comparison"
    rows = list(
        csv.DictReader((tmp_path / "cache/observations.csv").open(encoding="utf-8"))
    )
    perplexity = next(row for row in rows if row["metric_id"] == "train_perplexity")
    assert perplexity["implemented_availability"] == "available"
    assert perplexity["implemented_value"] == "120.0"
    assert perplexity["original_availability"] == "unavailable"
    assert perplexity["original_value"] == ""


def test_cli_variants_share_the_normalized_analysis_path(
    tmp_path: Path, monkeypatch
) -> None:
    uri = f"sqlite:///{tmp_path / 'cli-mlflow.db'}"
    monkeypatch.setenv("MLFLOW_F1_URL", uri)
    client = MlflowClient(uri)
    canonical_id = client.create_experiment("deepscratch.ds2")
    common = {
        "run.type": "seed_trial",
        "experiment.id": "e03",
        "master_seed": "1",
        "protocol.version": "book-source-v1",
        "runtime.device": "cuda:0",
    }
    implemented = client.create_run(
        canonical_id,
        tags={
            **common,
            "condition.id": "LM-SMALL-RNN",
            "implementation.variant": "implemented",
            "result.schema.name": "ds2-implemented",
            "result.schema.version": "1",
            "result.durable_complete": "true",
        },
    )
    client.log_metric(implemented.info.run_id, "final/train/perplexity", 120.0)
    client.set_terminated(implemented.info.run_id)
    original = client.create_run(
        canonical_id,
        tags={
            **common,
            "condition.id": "SMALL-RNNLM",
            "implementation.variant": "original",
            "result.schema.name": "ds2-original",
            "result.schema.version": "1",
        },
    )
    client.log_metric(original.info.run_id, "final/train/perplexity", 130.0)
    client.set_terminated(original.info.run_id)

    cli = CliRunner()
    for variant in ("implemented", "original", "all"):
        output = tmp_path / variant
        result = cli.invoke(
            app,
            [
                "analyze",
                "deepscratch",
                "ds2",
                "-e",
                "03",
                "--variant",
                variant,
                "--tracking-uri",
                uri,
                "--output-dir",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        labels = (
            ("imp", "org")
            if variant == "all"
            else ({"implemented": "imp", "original": "org"}[variant],)
        )
        for label in labels:
            summary = output / f"ds2_e03_{label}.md"
            assert summary.is_file()
            text = summary.read_text(encoding="utf-8")
            for metric in (
                "train_perplexity",
                "test_perplexity",
                "training_time_s",
                "parameter_count",
            ):
                assert f"| {metric} |" in text
