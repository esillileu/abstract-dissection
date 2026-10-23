from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from typer.testing import CliRunner
from w2v import NnlmDownpourTrainingSession, NnlmTrainingSession

from f2.catalog.manifest import read_manifest
from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.nnlm_artifacts import load_checkpoint
from f2.suites.w2v.tracked import run_tracked_yaml
from f2.suites.w2v1 import table3
from f2.suites.w2v1.executor import create_session
from repro_core.cli import app
from repro_core.context import RuntimePaths
from repro_core.execution.definition import RunOptions, RunSelection
from repro_core.execution.planning import Planner

CONDITION = "nnlm-d640-h640-n8-w320m"
DEFINITION_W2V = DEFINITION.get_suite("w2v1")
SOURCE = DEFINITION_W2V.config_root / "e02_table3.yaml"


def overrides():
    return {
        "vocabulary": {
            "initial_capacity": 16,
            "hash_capacity": 128,
            "min_count": 1,
            "max_lexical_words": 30,
        },
        "training": {
            "embedding_dimension": 4,
            "hidden_dimension": 5,
            "history_length": 2,
            "epochs": 2,
            "thread_count": 1,
            "observation_interval": 2,
        },
        "distribution": {"mini_batch_targets": 2, "parameter_server_shards": 2},
    }


def test_table3_plan_catalog_and_existing_config_parity():
    plans = Planner(DEFINITION_W2V).build(
        RunSelection(experiment_ids=("02",)), RunOptions()
    )
    manifest = read_manifest(Path("studies/f2/catalog/w2v.json"))
    slots = {x["planned_run_slot_id"]: x for x in manifest["planned_run_slots"]}
    assert len(plans) == 27
    nnlm = [p for p in plans if CONDITION in p.atomic_run_id]
    assert len(nnlm) == 9
    for plan in nnlm:
        spec = DEFINITION_W2V.load_run_spec(
            plan.path, atomic_run_id=plan.atomic_run_id, overrides={}
        ).with_seed(plan.seed)
        config = spec.to_executor_config()
        identity = config["identity"]
        slot = slots[identity["planned_run_slot_id"]]
        assert slot["seed"] == plan.seed
        assert slot["atomic_run_id"] == CONDITION
        assert identity["experiment_spec_id"] == "w2v1-table3-nnlm"
        assert config["corpus"]["lexical_token_budget"] == 320_000_000
        assert config["vocabulary"]["max_lexical_words"] == 82_000
        assert config["training"]["hidden_dimension"] == 640
        assert config["training"]["history_length"] == 8
        assert config["training"]["hidden_activation"] == "tanh"
        assert config["distribution"]["mode"] == "downpour"
    for architecture, digest in (
        ("cbow", "652e9f2c731108185e20710a13dda1e613cfe459c65d09b68b6e749965473287"),
        (
            "skipgram",
            "6ed1be753365ed8ecac880c79c5034eca36b4dfde1e7c3640fe5e21200700f8c",
        ),
    ):
        spec = DEFINITION_W2V.load_run_spec(
            SOURCE, atomic_run_id=f"wmt--{architecture}-d640-w320m", overrides={}
        )
        assert spec.to_executor_config()["identity"]["config_digest"] == digest


@pytest.mark.parametrize("mode", ["none", "downpour"])
def test_nnlm_executor_dispatch_and_verified_corpus_identity(tmp_path, mode):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a b a c d a b\nb a c b d a")
    spec = DEFINITION_W2V.load_run_spec(
        SOURCE, atomic_run_id=f"lm1b--{CONDITION}", overrides=overrides()
    )
    config = spec.to_executor_config()
    config["corpus"] = {"path": str(corpus)}
    if mode == "none":
        config["distribution"] = {"mode": "none"}
    session = create_session(
        config, tmp_path, corpus_digest="verified-materializer-sha256"
    )
    assert isinstance(
        session, NnlmTrainingSession if mode == "none" else NnlmDownpourTrainingSession
    )
    session.train_epoch()
    state = session.export_state()
    assert state.corpus_digest == "verified-materializer-sha256"
    assert state.processed_tokens == 13


def test_tracked_nnlm_publishes_checkpoint_and_shared_lookup(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a b a c d a b\nb a c b d a")
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()

    def materialize(config, _paths, **kwargs):
        config["corpus"] = {"path": str(corpus), "sha256": digest}

    monkeypatch.setattr("f2.suites.w2v.tracked._materialize_corpus", materialize)
    monkeypatch.setenv("REPRO_STAGING_ROOT", str(tmp_path / "staging"))
    monkeypatch.setenv("REPRO_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    receipt = run_tracked_yaml(
        SOURCE,
        atomic_run_id=f"lm1b--{CONDITION}",
        executor_module=DEFINITION_W2V.executor_module,
        spec_module=DEFINITION_W2V.spec_module,
        tracking_uri=uri,
        overrides=overrides(),
    )
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(receipt.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["result.durable_complete"] == "true"
    uploaded = {x.path for x in client.list_artifacts(receipt.run_id)}
    assert {"checkpoints", "lookup", "result_manifest.json"} <= uploaded
    state = load_checkpoint(receipt.result.checkpoint)
    assert state.completed_epochs == 2
    assert state.optimizer_kind == "adagrad"
    assert state.corpus_digest == digest
    lookup = load_lookup_artifact(receipt.result.lookup)
    assert np.array_equal(lookup.embeddings, state.input_embeddings())
    assert lookup.manifest["format"] == "f2-w2v-lookup-v1"


def run(architecture, seed, corpus="lm1b", **tags):
    condition = table3._CONDITIONS[architecture]
    return SimpleNamespace(
        info=SimpleNamespace(run_id=f"{corpus}-{architecture}-{seed}"),
        data=SimpleNamespace(
            tags={
                "implementation.variant": f"{corpus}--{condition}",
                "experiment_spec.id": f"w2v1-table3-{architecture}",
                "result.durable_complete": "true",
                "seed": str(seed),
                "f2.planned_run_slot_id": f"{table3._PLANS[corpus]}-{condition}-s{seed}",
                **tags,
            }
        ),
    )


def test_table3_selection_requires_all_canonical_durable_seeds():
    runs = [
        run(arch, seed) for arch in ("cbow", "skipgram", "nnlm") for seed in (1, 7, 19)
    ]
    selected = table3.complete_conditions(runs, "lm1b")
    assert set(selected) == {"cbow", "skipgram", "nnlm"}
    assert table3.complete_conditions(runs, "wmt") == {}
    assert table3.complete_conditions(runs[:-1], "lm1b").keys() == {"cbow", "skipgram"}
    for invalid in (
        {"result.durable_complete": "false"},
        {"experiment_spec.id": "w2v1-google-news-nnlm-6b"},
        {"f2.planned_run_slot_id": "wrong"},
    ):
        assert (
            table3.complete_conditions(
                [run("nnlm", 1), run("nnlm", 7), run("nnlm", 19, **invalid)], "lm1b"
            )
            == {}
        )
    duplicate = run("nnlm", 1)
    duplicate.info.run_id = "latest"
    assert (
        table3.complete_conditions([duplicate, *runs], "lm1b")["nnlm"][1].info.run_id
        == "latest"
    )


def test_msr_mirror_validation_and_download_contract(tmp_path, monkeypatch):
    content = b": all\n" + b"a b c d\n" * 8000
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(table3, "MSR_SYNTACTIC_SHA256", digest)
    target = tmp_path / "MSR-syntax.txt"
    downloader = Mock()
    downloader.download.side_effect = lambda url, path, **kwargs: path.write_bytes(
        content
    )
    monkeypatch.setattr(table3, "SerialDownloader", lambda: downloader)
    assert table3.ensure_msr_syntactic(target) == target
    downloader.download.assert_called_once_with(
        table3.MSR_SYNTACTIC_URL, target, expected_sha256=digest, expected_length=231634
    )
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        table3.ensure_msr_syntactic(target)
    with pytest.raises(ValueError, match="do not exist"):
        table3.ensure_msr_syntactic(tmp_path / "missing.txt")


def test_table3_analyzer_uses_both_benchmarks_and_shared_evaluator(
    tmp_path, monkeypatch
):
    from mlflow import MlflowClient
    from test_w2v_evaluation import Lookup

    words = tmp_path / "questions.txt"
    words.write_bytes(
        b": family\n" + b"a b c d\n" * 8869 + b": gram1\n" + b"a b c d\n" * 10675
    )
    msr = tmp_path / "msr.txt"
    msr.write_bytes(b": all\n" + b"a b c d\n" * 8000)
    monkeypatch.setattr(table3, "ensure_questions_words", lambda path: path)
    monkeypatch.setattr(table3, "ensure_msr_syntactic", lambda path: path)
    runs = [
        run(arch, seed) for arch in ("cbow", "skipgram", "nnlm") for seed in (1, 7, 19)
    ]
    monkeypatch.setattr(
        MlflowClient,
        "get_experiment_by_name",
        lambda self, name: SimpleNamespace(experiment_id="fixture"),
    )
    monkeypatch.setattr(MlflowClient, "search_runs", lambda self, *args, **kwargs: runs)
    metrics = tmp_path / "observations.csv"
    metrics.write_text("epoch,elapsed_seconds\n1,2.0\n2,3.0\n")
    monkeypatch.setattr(
        table3,
        "MlflowArtifactCache",
        lambda *args, **kwargs: SimpleNamespace(get=lambda run_id, artifact: metrics),
    )
    lookup = Lookup({b"a": (1, 0), b"b": (1, 1), b"c": (2, 0), b"d": (2, 1)})
    monkeypatch.setattr(table3, "load_lookup_artifact", lambda path: lookup)
    paths = RuntimePaths(
        repo_root=tmp_path,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
        staging_root=tmp_path / "staging",
        references_root=tmp_path / "references",
        studies_root=tmp_path / "studies",
    )
    outputs = table3.analyze_table3_sources(
        "http://localhost:5000",
        words,
        msr_questions_path=msr,
        corpus_source="lm1b",
        paths=paths,
    )
    with (outputs[0] / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert {row["architecture"] for row in rows} == {"cbow", "skipgram", "nnlm"}
    assert all(float(row["msr_accuracy_percent"]) == 100.0 for row in rows)
    assert all(
        int(row["semantic_total"]) == 8869 and int(row["syntactic_total"]) == 10675
        for row in rows
    )
    assert all(int(row["msr_included"]) == 8000 for row in rows)
    assert all(float(row["observed_training_seconds"]) == 5 for row in rows)
    assert "23 / 53 / 47" in (outputs[0] / "summary.md").read_text()


def test_cli_dispatches_table3_and_msr_path(tmp_path, monkeypatch):
    callback = Mock(return_value=(tmp_path / "table3",))
    monkeypatch.setattr(table3, "analyze_table3_sources", callback)
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "f2",
            "w2v1",
            "--table",
            "3",
            "--corpus",
            "lm1b",
            "--questions",
            str(tmp_path / "words.txt"),
            "--msr-questions",
            str(tmp_path / "msr.txt"),
            "--tracking-uri",
            "http://localhost:5000",
        ],
    )
    assert result.exit_code == 0, result.output
    assert callback.call_args.kwargs["msr_questions_path"] == tmp_path / "msr.txt"
    assert callback.call_args.kwargs["corpus_source"] == "lm1b"
