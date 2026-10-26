"""Paper reporting is deterministic, read-only, and explicit about missing evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from f2.suites.w2v.artifacts import _LOOKUP_ARRAYS, LOOKUP_FORMAT, _write_manifest
from f2.suites.w2v.evaluation import AnalogyQuestion
from f2.suites.w2v2 import analysis, paper
from f2.suites.w2v2.spec import parse_run_spec
from repro_core.context import RuntimePaths


def paths_at(root):
    return RuntimePaths(
        repo_root=root,
        data_root=root / "data",
        artifacts_root=root / "artifacts",
        cache_root=root / "cache",
        staging_root=root / "staging",
        references_root=root / "references",
        studies_root=root / "studies",
    )


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def refresh(record):
    """Test-owned stored inventory, using the existing result-manifest format."""
    from repro_mlflow.schema_v1.manifest import write_result_manifest

    root = Path(record["artifact_root"])
    (root / "result.json").write_text(json.dumps(record["result"], sort_keys=True))
    write_result_manifest(root)


def fixture_record(root, table, variant, seed=1):
    import shutil

    from f2.suites.w2v.artifacts import (
        _CHECKPOINT_ARRAYS,
        _VOCABULARY_ARRAYS,
        CHECKPOINT_FORMAT,
        VOCABULARY_FORMAT,
        _json_digest,
        _vocabulary_policy,
        training_config_digest,
    )
    from f2.suites.w2v.phrases import PhrasePolicy, _digest

    target = next(t for t in analysis.targets() if t[0] == table and t[3] == variant)
    config = (
        parse_run_spec(target[4]["_path"], atomic_run_id=variant)
        .with_seed(seed)
        .to_executor_config()
    )
    identity = config["identity"]
    directory = root / f"{table}-{variant}-{seed}"
    checkpoint = directory / "checkpoints/generations/final"
    checkpoint.mkdir(parents=True)
    tokens = sorted(
        {
            *paper.TABLE4,
            *paper.TABLE6_QUERIES,
            *" ".join(paper.TABLE5).split(),
            "answer",
            "a",
            "b",
            "c",
            "d",
        }
    )
    encoded = [t.encode() for t in tokens]
    vectors = (
        np.random.default_rng(12)
        .normal(size=(len(tokens), config["training"]["embedding_dimension"]))
        .astype(np.float32)
    )
    arrays = {
        "input_embeddings.npy": vectors,
        "output_embeddings.npy": np.zeros_like(vectors),
        "counts.npy": np.full(len(tokens), 6, dtype=np.uint64),
        "token_bytes.npy": np.frombuffer(b"".join(encoded), dtype=np.uint8),
        "token_offsets.npy": np.array(
            [0, *np.cumsum([len(t) for t in encoded])], dtype=np.uint64
        ),
        "huffman_offsets.npy": np.zeros(len(tokens) + 1, dtype=np.uint64),
        "huffman_paths.npy": np.zeros(0, dtype=np.uint64),
        "huffman_bits.npy": np.zeros(0, dtype=np.uint8),
    }
    for name, value in arrays.items():
        np.save(checkpoint / name, value, allow_pickle=False)
    source, transformed, vocab = "a" * 64, "b" * 64, "fnv1a64:0123456789abcdef"
    if table == "1":
        transformed = source
    state = {
        "schema_version": 1,
        "config_digest": training_config_digest(config["training"], seed),
        "completed_epochs": 3,
        "processed_tokens": 300,
        "corpus_digest": transformed,
        "vocabulary_digest": vocab,
        "vocabulary": {
            "hash_capacity": config["vocabulary"]["hash_capacity"],
            "retained_token_count": len(tokens) * 6,
        },
        "workers": [],
    }
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    _write_manifest(checkpoint, CHECKPOINT_FORMAT, state, _CHECKPOINT_ARRAYS)
    large = config.get("large_scale")
    lookup = checkpoint if large else directory / "lookup"
    if not large:
        lookup.mkdir()
        for name in _LOOKUP_ARRAYS:
            shutil.copyfile(checkpoint / name, lookup / name)
        _write_manifest(lookup, LOOKUP_FORMAT, state, _LOOKUP_ARRAYS)
    vocabulary = directory / "vocabulary"
    vocabulary.mkdir()
    for name in _VOCABULARY_ARRAYS:
        shutil.copyfile(checkpoint / name, vocabulary / name)
    _write_manifest(
        vocabulary,
        VOCABULARY_FORMAT,
        {
            "corpus_digest": transformed,
            "vocabulary_digest": vocab,
            "vocabulary_config": _vocabulary_policy(config["vocabulary"], table != "1"),
            "vocabulary_config_digest": _json_digest(
                _vocabulary_policy(config["vocabulary"], table != "1")
            ),
            "hash_capacity": config["vocabulary"]["hash_capacity"],
            "retained_token_count": len(tokens) * 6,
        },
        _VOCABULARY_ARRAYS,
    )

    def write(name, payload):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True))

    write(
        "execution_identity.json",
        {
            "seed": seed,
            "config_digest": identity["config_digest"],
            "planned_run_slot_id": identity["planned_run_slot_id"],
            "corpus_sha256": transformed,
            "run_id": directory.name,
        },
    )
    words = config["corpus"]["lexical_token_budget"]
    write(
        "data/dataset_manifest.json",
        {
            "resource_id": f"f2-{variant.split('--')[0]}-normalized",
            "resource_version_id": "test-registered-v1",
            "is_verified": True,
            "corpus_sha256": source,
            "lexical_tokens": words,
        },
    )
    write(
        "environment/system.json",
        {"machine": "x86_64", "processor": "test-cpu", "platform": "test-platform"},
    )
    phrase_tokens = words - 2
    if table != "1":
        policy = PhrasePolicy.from_config(config["phrase_detection"]).record()
        phrase = {
            "format": "f2-phrase-corpus-v2",
            "source_sha256": "d" * 64 if large else source,
            "corpus_sha256": transformed,
            "policy": policy,
            "policy_digest": _digest(policy),
            "token_count": phrase_tokens,
            "joined_count": 2,
            "pass_statistics": [
                {
                    "pass": 1,
                    "threshold": policy["thresholds"][0],
                    "corpus_sha256": "e" * 64,
                    "input_tokens": words,
                    "output_tokens": words - 1,
                    "joined_count": 1,
                },
                {
                    "pass": 2,
                    "threshold": policy["thresholds"][1],
                    "corpus_sha256": transformed,
                    "input_tokens": words - 1,
                    "output_tokens": phrase_tokens,
                    "joined_count": 1,
                },
            ],
        }
        phrase["lineage_digest"] = _digest(phrase)
        write("phrase_lineage.json", phrase)
    if large:
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        lineage = {
            **large,
            "seed": seed,
            "config_digest": identity["config_digest"],
            "planned_run_slot_id": identity["planned_run_slot_id"],
            "phrase_policy": config["phrase_detection"],
            "training_policy": config["training"],
            "vocabulary_policy": config["vocabulary"],
            "sentence_sha256": "d" * 64,
            "phrase_sha256": transformed,
            "lookup_manifest_sha256": hashlib.sha256(
                (lookup / "manifest.json").read_bytes()
            ).hexdigest(),
            "checkpoint_manifest_sha256": hashlib.sha256(
                (checkpoint / "manifest.json").read_bytes()
            ).hexdigest(),
            "vocabulary_artifact": {
                "digest": vocab,
                "files": {name: manifest["files"][name] for name in _VOCABULARY_ARRAYS},
            },
            "preflight": {
                "ready": True,
                "artifact_id": large["artifact_id"],
                "paper_lexical_tokens": large["paper_lexical_tokens"],
                "expected_lexical_tokens": words,
                "lexical_tokens": words,
                "sentence_policy": large["sentence_policy"],
                "execution_policy": large["execution_policy"],
                "source_sha256": source,
                "statistics": {
                    "vocabulary": {"words": len(tokens)},
                    "source_lexical_tokens": words,
                    "sentence_sha256": "d" * 64,
                    "phrase_sha256": transformed,
                    "sentence_distribution": {"lexical_tokens": phrase_tokens},
                },
            },
        }
        write("large_scale_lineage.json", lineage)
    metrics = directory / "metrics/observations.csv"
    metrics.parent.mkdir()
    metrics.write_text(
        "epoch,processed_tokens,elapsed_seconds\n1,100,60\n2,200,60\n3,300,60\n"
    )
    result = {
        "identity": identity,
        "complete": True,
        "epochs": [{"epoch": e, "processed_tokens": e * 100} for e in (1, 2, 3)],
        "epoch_history_complete": True,
        "artifacts": {
            "checkpoint": str(checkpoint.relative_to(directory)),
            "lookup": str(lookup.relative_to(directory)),
            "metrics": "metrics/observations.csv",
        },
    }
    record = {
        "run_id": directory.name,
        "artifact_root": str(directory),
        "status": "FINISHED",
        "lookup": str(lookup),
        "observations": str(metrics),
        "result": result,
        "tags": {
            "suite.name": "w2v2",
            "paper.id": "mikolov-2013-distributed-representations",
            "implementation.variant": variant,
            "seed": str(seed),
            "result.durable_complete": "true",
            "experiment_spec.id": identity.get(
                "experiment_spec_id", "w2v2-phrase-skipgram-1b-objectives"
            ),
            "f2.config_digest": identity["config_digest"],
            "f2.planned_run_slot_id": identity["planned_run_slot_id"],
            "f2.source_corpus_sha256": source,
        },
    }
    refresh(record)
    return record


def test_paper_numbers():
    assert list(paper.TABLE1.values()) == [
        (38, 63, 54, 59),
        (97, 63, 58, 61),
        (41, 53, 40, 47),
        (38, 60, 45, 53),
        (14, 61, 58, 60),
        (36, 61, 61, 61),
        (21, 52, 59, 55),
    ]
    assert paper.TABLE3 == {
        "neg5-no-subsampling": 24,
        "neg5-subsampling": 27,
        "neg15-no-subsampling": 27,
        "neg15-subsampling": 42,
        "hs-no-subsampling": 19,
        "hs-subsampling": 47,
    }
    assert paper.SECTION4 == {"phrase-6b": 66, "phrase-33b": 72}
    assert len(paper.TABLE2) == 10
    assert len(paper.TABLE4) == len(paper.TABLE5) == 5
    assert len(paper.TABLE6) == 4


def test_statistics():
    assert analysis.summarize([])["mean"] is None
    assert analysis.summarize([1])["ci95_low"] is None
    result = analysis.summarize([1, 2, 3])
    assert result["mean"] == 2 and result["standard_deviation"] == 1
    assert result["ci95_high"] == pytest.approx(2 + 4.30265273 / np.sqrt(3))


def test_missing_report_is_deterministic(tmp_path):
    paths = paths_at(tmp_path)
    output = analysis.generate_report(paths=paths)
    first = snapshot(output)
    analysis.generate_report(paths=paths)
    assert snapshot(output) == first
    payload = json.loads((output / "summary.json").read_text())
    assert payload["experiment_complete"] is False
    assert all(r["status"] == "no_data" for r in payload["runs"])
    assert all(a["missing_seeds"] == [1, 7, 19] for a in payload["aggregates"])
    for table in range(1, 7):
        assert f"## Table {table}" in (output / "summary.md").read_text()


def test_synthetic_all_tables_read_only_and_reused(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    records = [
        fixture_record(inputs, "1", "wmt--neg15-subsampling"),
        fixture_record(inputs, "3", "wmt--neg15-subsampling"),
        fixture_record(inputs, "3", "wmt--hs-subsampling"),
        fixture_record(inputs, "section4.1", "fineweb--phrase-33b"),
    ]
    questions = (
        AnalogyQuestion("test", b"a", b"b", b"c", b"d"),
        AnalogyQuestion("gram-test", b"a", b"b", b"c", b"d"),
        AnalogyQuestion("test", b"a", b"b", b"c", b"missing"),
        AnalogyQuestion("gram-test", b"a", b"b", b"c", b"missing"),
    )
    monkeypatch.setattr(
        analysis, "_dataset", lambda *args: ({"status": "complete"}, questions)
    )
    calls = []
    from f2.suites.w2v import artifacts

    original = artifacts.load_lookup_artifact

    def load(lookup, **kwargs):
        calls.append(lookup)
        return original(lookup, **kwargs)

    monkeypatch.setattr(artifacts, "load_lookup_artifact", load)
    before = snapshot(inputs)
    output = analysis.generate_report(records=records, paths=paths_at(tmp_path))
    payload = json.loads((output / "summary.json").read_text())
    assert len(calls) == 4
    assert {q["table"] for q in payload["qualitative"]} == {"4", "5", "6"}
    measured = [q for q in payload["qualitative"] if q["checkpoint_reused"]]
    assert len(measured) == 20
    assert all(q["checkpoint_reused"] for q in measured)
    assert all(len(q["neighbors"]) == 4 for q in measured if q["table"] == "5")
    complete = [r for r in payload["runs"] if r["status"] == "complete"]
    assert len(complete) == 4 and all(
        r["metrics"]["total"]["coverage"] == 0.5 for r in complete
    )
    assert snapshot(inputs) == before
    first = snapshot(output)
    analysis.generate_report(records=reversed(records), paths=paths_at(tmp_path))
    assert snapshot(output) == first


@pytest.mark.parametrize("failure", ["lookup", "duplicate", "unfinished", "oov"])
def test_invalid_and_incomplete(tmp_path, monkeypatch, failure):
    record = fixture_record(tmp_path / "inputs", "3", "wmt--hs-subsampling")
    questions = (AnalogyQuestion("test", b"absent", b"b", b"c", b"d"),)
    monkeypatch.setattr(
        analysis, "_dataset", lambda *args: ({"status": "complete"}, questions)
    )
    if failure == "lookup":
        record["result"]["artifacts"]["lookup"] += "-missing"
    elif failure == "unfinished":
        record["status"] = "RUNNING"
    refresh(record)
    records = [record, record] if failure == "duplicate" else [record]
    output = analysis.generate_report(records=records, paths=paths_at(tmp_path))
    result = next(
        r
        for r in json.loads((output / "summary.json").read_text())["runs"]
        if r.get("run_id")
    )
    assert result["status"] == (
        "running"
        if failure == "unfinished"
        else "incomplete"
        if failure == "oov"
        else "unverified"
        if failure == "lookup"
        else "failed"
    )
    if failure == "oov":
        assert result["metrics"]["total"]["coverage"] == 0


def test_dataset_integrity(tmp_path):
    assert analysis._dataset(None, True)[0]["status"] == "no_data"
    data = tmp_path / "questions.txt"
    data.write_text(": example\na b c d\n")
    assert analysis._dataset(data, True)[0]["status"] == "failed"
    assert analysis._dataset(data, False)[0]["status"] == "failed"


def test_three_seeds_and_missing_evaluation(tmp_path, monkeypatch):
    records = [
        fixture_record(tmp_path / "inputs", "3", "wmt--hs-subsampling", seed)
        for seed in analysis.SEEDS
    ]
    questions = (AnalogyQuestion("test", b"a", b"b", b"c", b"d"),)
    monkeypatch.setattr(
        analysis, "_dataset", lambda *args: ({"status": "complete"}, questions)
    )
    output = analysis.generate_report(records=records, paths=paths_at(tmp_path))
    payload = json.loads((output / "summary.json").read_text())
    aggregate = next(
        a
        for a in payload["aggregates"]
        if a["table"] == "3"
        and a["corpus"] == "wmt"
        and a["condition"] == "hs-subsampling"
    )
    assert aggregate["n"] == 3 and aggregate["missing_seeds"] == []
    assert aggregate["ci95_low"] is not None
    monkeypatch.setattr(analysis, "_dataset", lambda *args: ({"status": "no_data"}, ()))
    analysis.generate_report(records=records, paths=paths_at(tmp_path))
    payload = json.loads((output / "summary.json").read_text())
    assert all(r["status"] == "incomplete" for r in payload["runs"] if r.get("run_id"))


def test_cli_offline(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from f2.cli import app

    monkeypatch.setattr(RuntimePaths, "from_environment", lambda: paths_at(tmp_path))
    result = CliRunner().invoke(app, ["w2v2-report", "--offline"])
    assert result.exit_code == 0, result.output
    assert (paths_at(tmp_path).analysis_output("f2", "w2v2") / "summary.json").is_file()


@pytest.fixture
def canonical_questions(monkeypatch):
    questions = (
        AnalogyQuestion("capital-common-countries", b"a", b"b", b"c", b"d"),
        AnalogyQuestion("gram1-adjective-to-adverb", b"a", b"b", b"c", b"d"),
    )
    monkeypatch.setattr(
        analysis, "_dataset", lambda *args: ({"status": "complete"}, questions)
    )
    return questions


def report(tmp_path, records):
    output = analysis.generate_report(records=records, paths=paths_at(tmp_path))
    return json.loads((output / "summary.json").read_text())


def stored_json(record, name, mutate):
    path = Path(record["artifact_root"]) / name
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data, sort_keys=True))
    refresh(record)


def selected_aggregate(
    payload, metric="total", table="1", corpus="wmt", condition="neg15-subsampling"
):
    return next(
        a
        for a in payload["aggregates"]
        if (a["table"], a["corpus"], a["condition"], a["metric"])
        == (table, corpus, condition, metric)
    )


def test_all_tables_require_qualitative_evidence(
    tmp_path, monkeypatch, canonical_questions
):
    records = [
        fixture_record(tmp_path / "inputs", t, v, seed)
        for t, _, _, v, _ in analysis.targets()
        for seed in analysis.SEEDS
    ]
    before = snapshot(tmp_path / "inputs")
    payload = report(tmp_path, records)
    assert payload["report_generated"] is True
    assert payload["experiment_complete"] is True
    assert all(s == "complete" for s in payload["table_status"].values())
    monkeypatch.setattr(analysis, "nearest_tokens", lambda *args, **kwargs: ())
    monkeypatch.setattr(analysis, "additive_composition", lambda *args, **kwargs: ())
    payload = report(tmp_path, records)
    assert payload["report_generated"] is True
    assert payload["experiment_complete"] is False
    assert payload["status"] != "complete"
    assert len(payload["qualitative"]) == 120
    assert all(q["status"] == "no_data" for q in payload["qualitative"])
    assert all(payload["table_status"][t] != "complete" for t in ("4", "5", "6"))
    assert snapshot(tmp_path / "inputs") == before


def test_zero_valid_syntactic_is_missing_and_excluded(tmp_path, monkeypatch):
    records = [
        fixture_record(tmp_path / "inputs", "1", "wmt--neg15-subsampling", seed)
        for seed in analysis.SEEDS
    ]
    questions = (
        AnalogyQuestion("capital", b"a", b"b", b"c", b"d"),
        AnalogyQuestion("gram1-test", b"a", b"b", b"c", b"absent"),
    )
    monkeypatch.setattr(
        analysis, "_dataset", lambda *args: ({"status": "complete"}, questions)
    )
    payload = report(tmp_path, records)
    for row in payload["runs"]:
        if row.get("run_id"):
            metric = row["metrics"]["syntactic"]
            assert metric == {
                "status": "no_data",
                "accuracy_percent": None,
                "coverage": 0.0,
                "total_count": 1,
                "valid_count": 0,
                "correct_count": 0,
                "excluded_count": 1,
            }
            assert row["syntactic"] is None
    assert selected_aggregate(payload, "syntactic")["n"] == 0
    assert selected_aggregate(payload, "syntactic")["mean"] is None
    assert selected_aggregate(payload)["n"] == 3


def test_identical_embeddings_with_independent_evidence_are_valid(
    tmp_path, canonical_questions
):
    records = [
        fixture_record(tmp_path / "inputs", "1", "wmt--neg15-subsampling", seed)
        for seed in analysis.SEEDS
    ]
    # Every fixture deliberately has identical array bytes; distinct execution/run
    # evidence, not the numerical outcome, determines independence.
    payload = report(tmp_path, records)
    assert selected_aggregate(payload)["n"] == 3
    assert selected_aggregate(payload)["missing_seeds"] == []
    assert selected_aggregate(payload)["standard_deviation"] == 0


def test_partial_epoch_time_does_not_discard_accuracy(tmp_path, canonical_questions):
    record = fixture_record(tmp_path / "inputs", "1", "wmt--neg15-subsampling")
    Path(record["observations"]).write_text(
        "epoch,processed_tokens,elapsed_seconds\n1,100,60\n"
    )
    refresh(record)
    payload = report(tmp_path, [record])
    row = next(r for r in payload["runs"] if r.get("run_id"))
    assert row["identity_status"] == "complete"
    assert row["timing"]["status"] == "unverified"
    assert row["training_minutes"] is None
    assert selected_aggregate(payload)["n"] == 1
    assert selected_aggregate(payload, "training_minutes")["n"] == 0


def test_approved_33b_r3_preserves_distinct_counts(tmp_path, canonical_questions):
    record = fixture_record(tmp_path / "inputs", "section4.1", "fineweb--phrase-33b")
    row = next(r for r in report(tmp_path, [record])["runs"] if r.get("run_id"))
    assert row["identity_status"] == "complete"
    provenance = row["provenance"]
    assert provenance["actual_lexical_words"] == 32_991_052_431
    assert provenance["paper_lexical_words"] == 33_000_000_000
    assert provenance["phrase_tokens"] == 32_991_052_429
    assert provenance["processed_tokens_by_epoch"] == [100, 200, 300]
    assert provenance["execution_plan_id"].endswith("-r3")


@pytest.mark.parametrize(
    "state, expected",
    [
        ("FAILED", "failed"),
        ("KILLED", "killed"),
        ("RUNNING", "running"),
        ("FINISHED", "complete"),
    ],
)
def test_mlflow_states_are_distinct(tmp_path, canonical_questions, state, expected):
    record = fixture_record(tmp_path / "inputs", "3", "wmt--hs-subsampling")
    record["status"] = state
    record["tags"]["result.durable_complete"] = "false"
    row = next(r for r in report(tmp_path, [record])["runs"] if r.get("run_id"))
    assert row["status"] == expected


def test_repeated_run_id_cannot_supply_multiple_seeds(tmp_path, canonical_questions):
    records = [
        fixture_record(tmp_path / "inputs", "1", "wmt--neg15-subsampling", seed)
        for seed in analysis.SEEDS
    ]
    for record in records[1:]:
        record["run_id"] = records[0]["run_id"]
    assert selected_aggregate(report(tmp_path, records))["n"] == 0


def test_time_resume_deduplication_and_reset(tmp_path, canonical_questions):
    record = fixture_record(tmp_path / "inputs", "1", "wmt--neg15-subsampling")
    metrics = Path(record["observations"])
    metrics.write_text(
        "epoch,processed_tokens,elapsed_seconds\n1,100,60\n1,100,60\n2,200,60\n3,300,60\n"
    )
    refresh(record)
    row = next(r for r in report(tmp_path, [record])["runs"] if r.get("run_id"))
    assert row["training_minutes"] == 3
    assert len(row["timing"]["observations"]) == 3
    assert all(
        point["hardware"]["processor"] == "test-cpu"
        for point in row["timing"]["observations"]
    )
    metrics.write_text(
        "epoch,processed_tokens,elapsed_seconds\n1,50,60\n1,100,5\n2,200,60\n3,300,60\n"
    )
    refresh(record)
    row = next(r for r in report(tmp_path, [record])["runs"] if r.get("run_id"))
    assert row["timing"]["status"] == "unverified"
    assert row["training_minutes"] is None


def test_table2_validation_and_protocol_consistent_in_all_formats(tmp_path):
    import csv

    output = analysis.generate_report(paths=paths_at(tmp_path))
    payload = json.loads((output / "summary.json").read_text())
    with (output / "table2.csv").open() as stream:
        validation = next(
            row
            for row in csv.DictReader(stream)
            if row["kind"] == "evaluation_validation"
        )
    assert validation["status"] == payload["datasets"]["phrase"]["status"] == "no_data"
    assert validation["reason"] == payload["datasets"]["phrase"]["reason"]
    markdown = (output / "summary.md").read_text()
    assert json.dumps(payload["datasets"]["phrase"], sort_keys=True) in markdown
    assert "L2-normalize each input" in markdown
    assert payload["evaluation_protocol"]["candidate_scope"].startswith(
        "full checkpoint"
    )


def test_hardware_difference_and_missing_hardware_are_explicit(
    tmp_path, canonical_questions
):
    records = [
        fixture_record(tmp_path / "inputs", "1", "wmt--neg15-subsampling", seed)
        for seed in analysis.SEEDS
    ]
    stored_json(
        records[1],
        "environment/system.json",
        lambda data: data.update(processor="another-cpu"),
    )
    payload = report(tmp_path, records)
    timing = selected_aggregate(payload, "training_minutes")
    assert timing["n"] == 3
    assert timing["hardware_comparable_across_seeds"] is False
    assert timing["paper_hardware_comparability"].startswith("unverified")
    (Path(records[2]["artifact_root"]) / "environment/system.json").unlink()
    refresh(records[2])
    payload = report(tmp_path, records)
    assert selected_aggregate(payload)["n"] == 3
    assert selected_aggregate(payload, "training_minutes")["n"] == 2
