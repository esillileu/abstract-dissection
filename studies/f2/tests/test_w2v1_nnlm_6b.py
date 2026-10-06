"""6B NNLM plan identity and shared Table 4/6 evaluation, without production IO."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from typer.testing import CliRunner

from f2.catalog.manifest import read_manifest
from f2.corpus.fineweb import register
from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.nnlm_artifacts import load_checkpoint
from f2.suites.w2v1 import table4, table6
from repro_core.cli import app
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.execution.definition import RunOptions, RunSelection
from repro_core.execution.planning import Planner
from repro_core.execution.runner import run_config


def test_nnlm_6b_planner_and_fineweb_materialization_use_existing_spec(
    tmp_path, monkeypatch
):
    definition = DEFINITION.get_suite("w2v1")
    plans = Planner(definition).build(
        RunSelection(experiment_ids=("03",)), RunOptions()
    )
    nnlm_plans = [p for p in plans if "--nnlm-" in p.atomic_run_id]
    assert len(nnlm_plans) == 9
    repository = Mock()
    monkeypatch.setattr(register, "CatalogRepository", lambda conn: repository)
    counts = register.register_execution_plan(MagicMock())
    assert counts["planned_slots_count"] == 93
    assert counts["experiments_count"] == 6
    slots = {
        call.kwargs["planned_run_slot_id"]: call.kwargs
        for call in repository.upsert_planned_run_slot.call_args_list
    }
    manifest = read_manifest(Path("studies/f2/catalog/w2v.json"))
    requirement_ids = {x["requirement_id"] for x in manifest["requirements"]}
    assert all(
        call.kwargs["requirement_id"] in requirement_ids
        for call in repository.bind_plan_requirement.call_args_list
    )
    for plan in nnlm_plans:
        config = (
            definition.load_run_spec(
                plan.path, atomic_run_id=plan.atomic_run_id, overrides={}
            )
            .with_seed(plan.seed)
            .to_executor_config()
        )
        identity = config["identity"]
        assert identity["experiment_spec_id"] == "w2v1-google-news-nnlm-6b"
        slot = slots[identity["planned_run_slot_id"]]
        assert slot["seed"] == plan.seed
        assert (
            slot["plan_experiment_id"]
            == "w2v1-fineweb-reconstruction-r1-w2v1-google-news-nnlm-6b"
        )
        assert slot["parameters"]["training"] == config["training"]
        assert slot["parameters"]["distribution"] == config["distribution"]
        assert config["training"]["hidden_dimension"] == 640
        assert config["training"]["history_length"] == 8
        assert config["corpus"]["lexical_token_budget"] == 6_000_000_000
        assert config["vocabulary"]["max_lexical_words"] == 1_000_000
    bindings = [
        c.kwargs
        for c in repository.bind_plan_requirement.call_args_list
        if "nnlm" in c.kwargs["plan_experiment_id"]
    ]
    assert bindings[0]["resource_version_id"] == "f2-fineweb-2013-news-normalized-v1"
    assert bindings[0]["binding_type"] == "substitute"
    assert len(bindings) == 2
    table6_plans = Planner(definition).build(
        RunSelection(experiment_ids=("05",)), RunOptions()
    )
    assert len(table6_plans) == 6
    assert all("nnlm" not in p.atomic_run_id for p in table6_plans)


def run(arch, seed, dimension=100, **overrides):
    condition = (
        f"nnlm-d{dimension}-w6000m" if arch == "nnlm" else f"{arch}-d1000-w6000m"
    )
    spec = (
        "w2v1-google-news-nnlm-6b"
        if arch == "nnlm"
        else f"w2v1-google-news-{arch}-scale"
    )
    return SimpleNamespace(
        info=SimpleNamespace(run_id=f"{arch}-{dimension}-{seed}"),
        data=SimpleNamespace(
            tags={
                "implementation.variant": f"fineweb--{condition}",
                "experiment_spec.id": spec,
                "seed": str(seed),
                "result.durable_complete": "true",
                "f2.planned_run_slot_id": f"w2v1-fineweb-reconstruction-r1-{condition}-s{seed}",
                **overrides,
            }
        ),
    )


def test_table6_reuses_exact_table4_100d_runs_and_rejects_wrong_identity():
    runs = [
        run("nnlm", seed, dimension)
        for dimension in (20, 50, 100)
        for seed in (1, 7, 19)
    ]
    runs += [run(arch, seed) for arch in ("cbow", "skipgram") for seed in (1, 7, 19)]
    selected4 = table4.complete_conditions(runs, "fineweb")
    assert set(selected4) == {"nnlm-d20", "nnlm-d50", "nnlm-d100"}
    selected6 = table6.complete_conditions(runs)
    assert set(selected6) == {"nnlm", "cbow", "skipgram"}
    assert selected6["nnlm"] == selected4["nnlm-d100"]
    for seed in (1, 7, 19):
        assert selected6["nnlm"][seed] is selected4["nnlm-d100"][seed]
    assert table6.complete_conditions(runs[:6]) == {}
    for tags in (
        {"result.durable_complete": "false"},
        {"experiment_spec.id": "w2v1-table3-nnlm"},
        {"f2.planned_run_slot_id": "new-table6-nnlm-run"},
    ):
        invalid = [run("nnlm", 1), run("nnlm", 7), run("nnlm", 19, **tags)]
        assert table6.complete_conditions(invalid) == {}


def test_table4_6_analysis_reference_runtime_and_shared_lookup(tmp_path, monkeypatch):
    import mlflow
    from test_w2v_evaluation import Lookup

    words = tmp_path / "words.txt"
    words.write_bytes(
        b": family\n" + b"a b c d\n" * 8869 + b": gram1\n" + b"a b c d\n" * 10675
    )
    runs = [
        run("nnlm", seed, dimension)
        for dimension in (20, 50, 100)
        for seed in (1, 7, 19)
    ]
    runs += [run(arch, seed) for arch in ("cbow", "skipgram") for seed in (1, 7, 19)]
    client = SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(experiment_id="fixture"),
        search_runs=lambda *args, **kwargs: runs,
    )
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **kwargs: client)
    metrics = tmp_path / "observations.csv"
    metrics.write_text("epoch,elapsed_seconds\n1,2.0\n2,3.0\n")
    cache = SimpleNamespace(get=lambda run_id, artifact: metrics)
    lookup = Lookup({b"a": (1, 0), b"b": (1, 1), b"c": (2, 0), b"d": (2, 1)})
    for module in (table4, table6):
        monkeypatch.setattr(module, "ensure_questions_words", lambda path: path)
        monkeypatch.setattr(
            module, "MlflowArtifactCache", lambda *args, **kwargs: cache
        )
        monkeypatch.setattr(module, "load_lookup_artifact", lambda path: lookup)
    paths = RuntimePaths(
        repo_root=tmp_path,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
        staging_root=tmp_path / "staging",
        references_root=tmp_path / "references",
        studies_root=tmp_path / "studies",
    )
    (out4,) = table4.analyze_table4_sources(
        "http://localhost:5000", words, corpus_source="fineweb", paths=paths
    )
    (out6,) = table6.analyze_table6_sources("http://localhost:5000", words, paths=paths)
    with (out4 / "runs.csv").open() as stream:
        rows4 = list(csv.DictReader(stream))
    with (out6 / "runs.csv").open() as stream:
        rows6 = list(csv.DictReader(stream))
    assert len(rows4) == len(rows6) == 9
    assert {
        row["mlflow_run_id"] for row in rows4 if row["architecture"] == "nnlm-d100"
    } == {row["mlflow_run_id"] for row in rows6 if row["architecture"] == "nnlm"}
    assert all(float(row["total_accuracy_percent"]) == 100 for row in rows4 + rows6)
    for row in rows6:
        assert float(row["observed_training_seconds"]) == 5
        if row["architecture"] == "nnlm":
            assert row["source_table"] == "4"
            assert row["paper_training_days"] == "14"
            assert row["paper_cpu_cores"] == "180"
    summary4 = (out4 / "summary.md").read_text()
    assert all(
        name in summary4 for name in ("Collobert-Weston", "Turian", "Mnih", "Huang")
    )
    assert "20.3" in summary4 and "43.2" in summary4 and "50.8" in summary4
    summary6 = (out6 / "summary.md").read_text()
    assert "14 x 180" in summary6
    assert "not measurements of the local environment" in summary6
    with pytest.raises(ValueError, match="FineWeb"):
        table6.analyze_table6_sources(
            "http://localhost:5000", words, corpus_source="wmt", paths=paths
        )


@pytest.mark.parametrize("table", (4, 6))
def test_cli_table4_6_accepts_fineweb(tmp_path, monkeypatch, table):
    module = table4 if table == 4 else table6
    callback = Mock(return_value=(tmp_path / f"table{table}",))
    monkeypatch.setattr(module, f"analyze_table{table}_sources", callback)
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "f2",
            "w2v1",
            "--table",
            str(table),
            "--corpus",
            "fineweb",
            "--tracking-uri",
            "http://localhost:5000",
        ],
    )
    assert result.exit_code == 0, result.output
    assert callback.call_args.kwargs["corpus_source"] == "fineweb"


def test_table4_nnlm_fixture_finishes_with_resumable_checkpoint(tmp_path):
    definition = DEFINITION.get_suite("w2v1")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a b a c d a b\nb a c b d a")
    paths = RuntimePaths(
        repo_root=tmp_path,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
        staging_root=tmp_path / "staging",
        references_root=tmp_path / "references",
        studies_root=tmp_path / "studies",
    )
    spec = definition.load_run_spec(
        definition.config_root / "e03_table4.yaml",
        atomic_run_id="fineweb--nnlm-d20-w6000m",
        overrides={
            "corpus": {"path": str(corpus)},
            "vocabulary": {
                "initial_capacity": 16,
                "hash_capacity": 128,
                "min_count": 1,
                "max_lexical_words": 30,
            },
            "training": {
                "hidden_dimension": 5,
                "history_length": 2,
                "epochs": 2,
                "thread_count": 1,
                "observation_interval": 2,
            },
            "distribution": {"mini_batch_targets": 2},
        },
    )
    result = run_config(
        spec.to_executor_config(),
        ExperimentContext(paths=paths),
        executor_module=definition.executor_module,
    )
    state = load_checkpoint(result.checkpoint)
    assert state.completed_epochs == 2
    assert state.optimizer_kind == "adagrad"
    assert state.input_embeddings().shape[1] == 20
    assert load_lookup_artifact(result.lookup).embeddings.shape[1] == 20


def test_existing_table4_config_digests_are_unchanged():
    definition = DEFINITION.get_suite("w2v1")
    for corpus in ("wmt", "lm1b", "umbc"):
        for architecture, digest in (
            (
                "cbow",
                "cf774c3827ddd30d0eef48420a400380ae808dce9c17993320016da38a2dba8d",
            ),
            (
                "skipgram",
                "8c0a02f43b414b11bb53da77f78c2f4776edd9502a65eaa67d7a107dfdecb0f2",
            ),
        ):
            config = definition.load_run_spec(
                definition.config_root / "e03_table4.yaml",
                atomic_run_id=f"{corpus}--{architecture}-d300-w783m",
                overrides={},
            ).to_executor_config()
            assert config["identity"]["config_digest"] == digest


def test_final_w2v1_table_training_matrix_has_no_duplicate_nnlm_100d_run():
    definition = DEFINITION.get_suite("w2v1")
    for experiment, count, architectures in (
        ("01", 216, {"cbow"}),
        ("02", 27, {"cbow", "skip_gram", "feedforward_nnlm"}),
        ("03", 27, {"cbow", "skip_gram", "feedforward_nnlm"}),
        ("04", 18, {"cbow", "skip_gram"}),
        ("05", 6, {"cbow", "skip_gram"}),
        ("06", 3, {"skip_gram"}),
    ):
        plans = Planner(definition).build(
            RunSelection(experiment_ids=(experiment,)), RunOptions()
        )
        assert len(plans) == count
        actual = {
            definition.load_run_spec(
                p.path, atomic_run_id=p.atomic_run_id, overrides={}
            ).to_executor_config()["training"]["model_kind"]
            for p in plans
        }
        assert actual == architectures
