import csv
from pathlib import Path

from mlflow.tracking import MlflowClient

from exp.deepscratch.cross_variant import write_comparison_table
from exp.deepscratch.identity import Volume


def test_comparison_table_marks_absent_native_metric_unavailable(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(uri)
    new_id = client.create_experiment("deepscratch.ds2")
    legacy_id = client.create_experiment("ds2_original")
    common = {
        "run.type": "seed_trial",
        "experiment.id": "e03",
        "atomic_run.id": "LM-SMALL-RNN",
        "master_seed": "1",
        "protocol.version": "book-source-v1",
    }
    implemented = client.create_run(
        new_id,
        tags={**common, "implementation.variant": "implemented"},
    )
    client.log_metric(implemented.info.run_id, "final/train/perplexity", 120.0)
    client.set_terminated(implemented.info.run_id)
    original = client.create_run(legacy_id, tags=common)
    client.set_terminated(original.info.run_id)

    output = write_comparison_table(
        uri,
        volume=Volume.DS2,
        experiment_ids=["e03"],
        output_dir=tmp_path / "comparison",
    )

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    perplexity = next(row for row in rows if row["metric_id"] == "train_perplexity")
    assert perplexity["implemented_availability"] == "available"
    assert perplexity["implemented_value"] == "120.0"
    assert perplexity["original_availability"] == "unavailable"
    assert perplexity["original_value"] == ""
