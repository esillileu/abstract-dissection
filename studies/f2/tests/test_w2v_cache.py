"""Finished MLflow artifacts remain analyzable without execution scratch."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.cache import LocalResults, download_results, read_records
from f2.suites.w2v.evaluation import AnalogyQuestion, evaluate_analogies


@pytest.mark.parametrize("suite", ["w2v1", "w2v2"])
def test_finished_results_survive_staging_and_cache_deletion(
    tmp_path, monkeypatch, suite
):
    from mlflow import MlflowClient
    from test_w2v2_analysis import fixture_record, paths_at

    # Existing-format artifacts, with no training or corpus acquisition.
    record = fixture_record(tmp_path / "staging", "1", "wmt--hs-subsampling")
    source = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = source.as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(f"f2.{suite}")
    artifact_root = Path(record["artifact_root"])
    benchmarks = artifact_root / "evaluation"
    benchmarks.mkdir()
    (benchmarks / "questions-words.txt").write_bytes(
        b": family\n" + b"a b c d\n" * 5 + b": gram1\n" + b"a b c d\n" * 5
    )
    run_ids = []
    for seed in (1, 7, 19) if suite == "w2v1" else (1,):
        tags = {**record["tags"], "seed": str(seed)}
        if suite == "w2v1":
            tags.update(
                {
                    "suite.name": suite,
                    "implementation.variant": "wmt--d50-w24m",
                    "experiment_spec.id": "w2v1-table2-cbow",
                }
            )
        tags.pop("result.durable_complete", None)
        run_id = client.create_run(experiment, tags=tags).info.run_id
        client.log_artifacts(run_id, str(artifact_root))
        client.set_terminated(run_id, "FINISHED")
        run_ids.append(run_id)
    run_id = run_ids[0]
    paths = paths_at(tmp_path)
    if paths.staging_root.exists():
        shutil.rmtree(paths.staging_root)
    questions = (
        AnalogyQuestion("family", b"a", b"b", b"c", b"d"),
        AnalogyQuestion("gram1", b"b", b"a", b"d", b"c"),
    )

    def analyze(root):
        local = LocalResults(root)
        artifact = json.loads(local.get(run_id, "result.json").read_text())
        lookup = load_lookup_artifact(
            local.get(run_id, artifact["artifacts"]["lookup"])
        )
        evaluation = asdict(evaluate_analogies(lookup, questions))
        assert evaluation["overall"]["valid_count"] == 2
        if suite == "w2v1":
            from f2.suites.w2v1.analysis import analyze_table2

            questions_path = local.get(run_id, "evaluation/questions-words.txt")
            output = analyze_table2(root, questions_path, paths=paths)
            report = json.loads((output / "table2-results.json").read_text())
            assert len(report["runs"]) == 3
        else:
            from f2.suites.w2v2 import analysis

            with monkeypatch.context() as patch:
                patch.setattr(
                    analysis,
                    "_dataset",
                    lambda *args: ({"status": "complete"}, questions),
                )
                output = analysis.generate_report(
                    records=read_records(root), paths=paths
                )
            report = json.loads((output / "summary.json").read_text())
            assert any(r["status"] == "complete" for r in report["runs"])
        return evaluation, report

    root = download_results(uri, suite, paths)
    assert read_records(root)[0]["status"] == "FINISHED"
    # Analyzer remains usable even when all MLflow access is forbidden.
    with monkeypatch.context() as patch:
        patch.setattr(
            "mlflow.MlflowClient", lambda **kw: pytest.fail("analyzer used MLflow")
        )
        first = analyze(root)
    shutil.rmtree(paths.cache_root)
    second = analyze(download_results(uri, suite, paths))
    assert second == first
    assert not paths.staging_root.exists()
    assert client.get_run(run_id).info.status == "FINISHED"


def test_missing_cached_artifact_does_not_query_or_recover(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "runs.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "finished",
                    "status": "FINISHED",
                    "tags": {},
                    "artifact_root": str(root / "finished"),
                }
            ]
        )
    )
    monkeypatch.setattr(
        "mlflow.MlflowClient", lambda **kw: pytest.fail("network access")
    )
    local = LocalResults(root)
    assert local.runs()[0].info.status == "FINISHED"
    with pytest.raises(FileNotFoundError):
        local.get("finished", "lookup")
    assert not (root / "finished").exists()


def test_analysis_modules_do_not_import_tracking_clients():
    import ast
    from pathlib import Path

    source = Path(__file__).parents[1] / "src/f2/suites"
    modules = [
        source / "w2v1" / f"{name}.py"
        for name in ("analysis", "table3", "table4", "table5", "table6", "table7")
    ]
    modules.append(source / "w2v2/analysis.py")
    for module in modules:
        for node in ast.walk(ast.parse(module.read_text())):
            names = (
                [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(n.startswith(("mlflow", "repro_mlflow")) for n in names), (
                module
            )
