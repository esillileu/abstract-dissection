"""Section 4.1 configuration/readiness and test-owned small 1000d HS runs."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from repro_io.checksum import sha256_file

from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import load_checkpoint
from f2.suites.w2v2.executor import W2V2Executor
from f2.suites.w2v2.preflight import cached_sentences, preflight, sentences
from f2.suites.w2v2.spec import parse_run_spec
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.execution.definition import RunOptions, RunSelection
from repro_core.execution.planning import Planner


def config(scale=6):
    definition = DEFINITION.get_suite("w2v2")
    number = "03" if scale == 6 else "04"
    path = definition.config_root / f"e{number}_section4_phrase_{scale}b.yaml"
    return (
        parse_run_spec(path, atomic_run_id=f"fineweb--phrase-{scale}b")
        .with_seed(1)
        .to_executor_config()
    )


@pytest.mark.parametrize("scale,experiment", [(6, "03"), (33, "04")])
def test_scale_parsing_planning_and_missing_source_preflight(
    tmp_path, scale, experiment
):
    resolved = config(scale)
    plans = Planner(DEFINITION.get_suite("w2v2")).build(
        RunSelection(experiment_ids=(experiment,)), RunOptions()
    )
    assert len(plans) == 3
    assert len({p.seed for p in plans}) == 3
    result = preflight(resolved, tmp_path / "missing", tmp_path)
    assert not result["ready"]
    assert (
        result["artifact_id"]
        == f"f2.w2v2.section4.phrase-{scale}b.hs1000.entire-sentence.r{2 if scale == 6 else 3}"
    )
    assert result["status"] == "blocked"
    assert "estimate" not in result
    assert config(6)["identity"] != config(33)["identity"]
    assert (
        config(6)["large_scale"]["artifact_id"]
        != config(33)["large_scale"]["artifact_id"]
    )


def test_nominal_33b_uses_accepted_exact_budget_and_distinct_identity(tmp_path):
    resolved = config(33)
    large = resolved["large_scale"]
    assert large["paper_lexical_tokens"] == 33_000_000_000
    assert large["lexical_tokens"] == 32_991_052_431
    assert resolved["corpus"]["lexical_token_budget"] == large["lexical_tokens"]
    report = preflight(resolved, tmp_path / "missing", tmp_path)
    assert report["lexical_shortfall"] == 8_947_569
    assert report["expected_lexical_tokens"] == 32_991_052_431
    path = DEFINITION.get_suite("w2v2").config_root / "e04_section4_phrase_33b.yaml"
    exact = (
        parse_run_spec(
            path,
            atomic_run_id="fineweb--phrase-33b",
            overrides={
                "large_scale": {
                    "lexical_tokens": 33_000_000_000,
                    "artifact_id": "f2.w2v2.section4.phrase-33b.hs1000.entire-sentence.r2",
                },
                "corpus": {"lexical_token_budget": 33_000_000_000},
            },
        )
        .with_seed(1)
        .to_executor_config()
    )
    assert exact["identity"]["config_digest"] != resolved["identity"]["config_digest"]


@pytest.mark.parametrize("count", [0, -1, 33_000_000_001, True])
def test_invalid_actual_33b_budget_is_rejected(count):
    path = DEFINITION.get_suite("w2v2").config_root / "e04_section4_phrase_33b.yaml"
    with pytest.raises(ValueError, match="paper/actual lexical"):
        parse_run_spec(
            path,
            atomic_run_id="fineweb--phrase-33b",
            overrides={"large_scale": {"lexical_tokens": count}},
        )


def test_sentence_reconstruction_cache_and_bounds(tmp_path):
    source = tmp_path / "documents.txt"
    source.write_bytes(b"First sentence. Another one!\nNo terminal punctuation\nLast?")
    expected = [
        [b"First", b"sentence."],
        [b"Another", b"one!"],
        [b"No", b"terminal", b"punctuation"],
        [b"Last?"],
    ]
    assert list(sentences(source)) == expected
    digest = sha256_file(source)
    path = cached_sentences(source, tmp_path, digest)
    assert (
        path.read_bytes()
        == b"First sentence.\nAnother one!\nNo terminal punctuation\nLast?\n"
    )
    assert cached_sentences(source, tmp_path, digest) == path
    path.write_bytes(b"corrupted")
    assert cached_sentences(source, tmp_path, digest).read_bytes().endswith(b"Last?\n")
    source.write_bytes(b"x " * 1001 + b"\n")
    assert len(next(sentences(source))) == 1001
    source.write_bytes(b"x " * 65537)
    with pytest.raises(ValueError, match="splitting is forbidden"):
        list(sentences(source))
    source.write_bytes(b"x" * 4097)
    with pytest.raises(ValueError, match="4096"):
        list(sentences(source))


def small_config(tmp_path):
    resolved = config()
    source = tmp_path / "source.txt"
    source.write_bytes(
        b"new york city has many people. new york state has many towns!\nnew york city has many people. new york state has many towns!\n"
    )
    resolved["corpus"] = {"path": str(source), "sha256": sha256_file(source)}
    resolved["large_scale"]["lexical_tokens"] = 24  # test-owned reduced input
    resolved["vocabulary"].update(initial_capacity=2, hash_capacity=101, min_count=1)
    resolved["training"].update(
        epochs=2, thread_count=2, subsampling_threshold=0.0, observation_interval=1
    )
    resolved["phrase_detection"].update(min_count=1, threshold=0.0)
    return resolved, source


def test_insufficient_resources_count_and_checksum_block_before_preparation(
    tmp_path, monkeypatch
):
    resolved, source = small_config(tmp_path)
    result = preflight(resolved, source, tmp_path, available_ram=1, available_disk=1)
    assert not result["ready"]
    assert any("RAM" in b for b in result["blockers"])
    result = preflight(
        resolved, source, tmp_path, available_ram=2**40, available_disk=1
    )
    assert any("disk" in b for b in result["blockers"])
    resolved["large_scale"]["lexical_tokens"] = 6_000_000_000
    resolved["corpus"]["sha256"] = "0" * 64
    result = preflight(resolved, source, tmp_path)
    assert any("SHA256 mismatch" in b for b in result["blockers"])
    resolved["corpus"]["sha256"] = sha256_file(source)
    result = preflight(resolved, source, tmp_path)
    assert any("expected exactly" in b for b in result["blockers"])
    monkeypatch.setattr(
        "f2.suites.w2v2.executor.cached_phrase_corpus",
        lambda *a, **kw: pytest.fail("phrase preparation started"),
    )
    with pytest.raises(ValueError, match="preflight blocked"):
        W2V2Executor().run(resolved, ExperimentContext(paths=paths(tmp_path)))


def paths(tmp_path):
    return dataclasses.replace(
        RuntimePaths.from_environment(),
        workspace_root=tmp_path / "workspace",
        cache_root=tmp_path / "cache",
        staging_root=tmp_path / "staging",
    )


def test_1000d_hs_phrase_training_saved_checkpoint_and_lineage(tmp_path):
    resolved, source = small_config(tmp_path)
    context = ExperimentContext(paths=paths(tmp_path))
    assert preflight(resolved, source, context.paths.cache_root, prepare=True)["ready"]
    result = W2V2Executor().run(resolved, context)
    state = load_checkpoint(result.checkpoint)
    assert state.completed_epochs == 2
    assert (state.output_embeddings() != 0).any()
    assert state.input_embeddings().shape[1] == 1000
    lineage = json.loads((result.root / "large_scale_lineage.json").read_text())
    assert lineage["artifact_id"] == resolved["large_scale"]["artifact_id"]
    assert lineage["preflight"]["source_sha256"] == sha256_file(source)
    assert lineage["checkpoint_manifest_sha256"] == sha256_file(
        result.checkpoint / "manifest.json"
    )
    assert lineage["phrase_sha256"] != lineage["sentence_sha256"]
    assert lineage["config_digest"] == resolved["identity"]["config_digest"]
    with pytest.raises(ValueError, match="not empty"):
        W2V2Executor().run(resolved, context)


def test_catalog_declares_both_scales_without_claiming_ready_data():
    from f2.catalog.manifest import read_manifest

    manifest = read_manifest(Path("studies/f2/catalog/w2v.json"))
    for scale in (6, 33):
        sid = f"w2v2-section4-phrase-{scale}b"
        assert any(
            row["experiment_spec_id"] == sid for row in manifest["experiment_specs"]
        )
        plan = next(
            row
            for row in manifest["execution_plans"]
            if row["execution_plan_id"]
            == f"w2v2-section4-{scale}b-reconstruction-r{2 if scale == 6 else 3}"
        )
        assert plan["status"] == "draft"
    assert not any(
        row["experiment_spec_id"] == "w2v2-large-phrase-skipgram"
        for row in manifest["experiment_specs"]
    )


def test_invalid_engine_policy_is_blocked_before_source_scan(tmp_path, monkeypatch):
    resolved, source = small_config(tmp_path)
    resolved["training"]["thread_count"] = 0
    monkeypatch.setattr(
        "f2.suites.w2v2.preflight.sentences",
        lambda *args: pytest.fail("invalid policy started source scan"),
    )
    result = preflight(resolved, source, tmp_path)
    assert not result["ready"]
    assert "invalid engine policy" in result["blockers"][0]


def test_measured_preflight_reuses_prepared_phrase_vocabulary_and_profiles(
    tmp_path, monkeypatch
):
    resolved, source = small_config(tmp_path)
    cache = tmp_path / "cache"
    first = preflight(resolved, source, cache)
    assert first["status"] == "requires_preparation"
    measured = preflight(resolved, source, cache, prepare=True)
    assert measured["ready"], measured
    stats = measured["statistics"]
    assert stats["source_lexical_tokens"] == 24
    assert stats["sentence_distribution"]["lexical_tokens"] < 24
    assert (
        measured["estimate"]["model_bytes"] == stats["vocabulary"]["words"] * 1000 * 8
    )
    assert measured["estimate"]["corpus_read_bytes"] == stats["phrase_bytes"] * 2
    assert (
        "longest_one_percent_pair_fraction" in stats["retained_sentence_distribution"]
    )
    monkeypatch.setattr(
        "f2.suites.w2v2.preflight._measure_phrase",
        lambda *a: pytest.fail("counted cached data again"),
    )
    resolved["identity"]["seed"] = 7
    again = preflight(resolved, source, cache)
    assert again["ready"]
    assert again["statistics"] == stats
    assert not list((cache / "f2/w2v/sentences").glob("*.txt"))


@pytest.mark.parametrize(
    "section,key",
    [("training", "sigmoid_table_size"), ("vocabulary", "initial_capacity")],
)
def test_preflight_counts_configured_reservations(tmp_path, section, key):
    resolved, source = small_config(tmp_path)
    resolved[section][key] = 10**12
    result = preflight(
        resolved,
        source,
        tmp_path,
        available_ram=2 * 1024**3,
        available_disk=10**12,
        prepare=True,
    )
    assert not result["ready"]
    assert "reservations" in result["blockers"][0]
    assert not (tmp_path / "f2/w2v").exists()


def test_cgroup_current_and_ancestor_limits_are_applied(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from f2.suites.w2v2.preflight import available_memory

    root = tmp_path / "cgroup"
    leaf = root / "user/session"
    leaf.mkdir(parents=True)
    proc = tmp_path / "proc"
    proc.write_text("0::/user/session\n")
    for path, maximum, current in (
        (root, "max", "0"),
        (root / "user", "1000", "300"),
        (leaf, "900", "50"),
    ):
        (path / "memory.max").write_text(maximum)
        (path / "memory.current").write_text(current)
    monkeypatch.setattr(
        "f2.suites.w2v2.preflight.psutil.virtual_memory",
        lambda: SimpleNamespace(available=2000),
    )
    assert available_memory(cgroup_root=root, proc_cgroup=proc) == 700


def test_chunked_checkpoint_never_exports_full_matrices_and_reuses_final(
    tmp_path, monkeypatch
):
    from w2v import TrainingSession

    from f2.suites.w2v import artifacts

    resolved, source = small_config(tmp_path)
    monkeypatch.setattr(
        TrainingSession,
        "export_state",
        lambda *a: pytest.fail("full matrix export used"),
    )
    monkeypatch.setattr(
        TrainingSession,
        "vocabulary_state",
        lambda *a: pytest.fail("copied cached vocabulary into a checkpoint"),
    )
    monkeypatch.setattr(artifacts, "MATRIX_CHUNK_ELEMENTS", 17)
    context = ExperimentContext(paths=paths(tmp_path))
    result = W2V2Executor().run(resolved, context)
    latest = json.loads((result.root / "checkpoints/latest.json").read_text())
    final = json.loads((result.root / "checkpoints/final.json").read_text())
    assert latest["path"] == final["path"]
    assert len(list((result.root / "checkpoints/generations").iterdir())) == 1
    assert (result.lookup / "input_embeddings.npy").stat().st_ino == (
        result.checkpoint / "input_embeddings.npy"
    ).stat().st_ino
    with pytest.raises(ValueError, match="not empty"):
        W2V2Executor().run(resolved, context)
    assert load_checkpoint(result.checkpoint).completed_epochs == 2


def test_phrase_cache_is_shared_across_vocabulary_conditions(tmp_path, monkeypatch):
    resolved, source = small_config(tmp_path)
    cache = tmp_path / "cache"
    first = preflight(resolved, source, cache, prepare=True)
    assert first["ready"]
    resolved["vocabulary"]["max_lexical_words"] = 2
    monkeypatch.setattr(
        "f2.suites.w2v2.preflight.cached_sentences",
        lambda *a: pytest.fail("reconstructed an existing phrase input"),
    )
    second = preflight(resolved, source, cache, prepare=True)
    assert second["ready"]
    assert second["statistics"]["vocabulary"]["words"] == 3
    assert second["statistics"]["phrase_path"] == first["statistics"]["phrase_path"]
    assert (
        second["statistics"]["vocabulary_path"]
        != first["statistics"]["vocabulary_path"]
    )


def test_interrupted_execution_preserves_scratch_without_resuming(
    tmp_path, monkeypatch
):
    from w2v import TrainingSession

    resolved, source = small_config(tmp_path)
    context = ExperimentContext(paths=paths(tmp_path))
    with monkeypatch.context() as patcher:

        def interrupt(*args):
            raise RuntimeError("test interruption before first epoch completes")

        patcher.setattr(TrainingSession, "train_epoch", interrupt)
        with pytest.raises(RuntimeError, match="interruption"):
            W2V2Executor().run(resolved, context)
    with pytest.raises(ValueError, match="not empty"):
        W2V2Executor().run(resolved, context)
    assert list(context.paths.staging_root.rglob("latest.json"))


def test_prepared_statistics_do_not_bypass_current_resource_gate(tmp_path, monkeypatch):
    resolved, source = small_config(tmp_path)
    context = ExperimentContext(paths=paths(tmp_path))
    resolved["prepared_training"] = preflight(
        resolved, source, context.paths.cache_root, prepare=True
    )
    assert resolved["prepared_training"]["ready"]
    monkeypatch.setattr("f2.suites.w2v2.preflight.available_memory", lambda: 1)
    monkeypatch.setattr(
        "f2.suites.w2v.executor.create_session",
        lambda *a, **kw: pytest.fail("started with stale resource readiness"),
    )
    with pytest.raises(ValueError, match=r"preflight blocked.*RAM"):
        W2V2Executor().run(resolved, context)
