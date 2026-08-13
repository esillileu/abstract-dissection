import ast
from pathlib import Path

from typer.testing import CliRunner

from exp.cli import PLUGIN_REGISTRY, app
from exp.deepscratch.cli import _writer_overrides
from exp.deepscratch.identity import DeepScratchCoordinate, Variant, Volume, legacy_namespace
from exp.deepscratch.comparison import ComparableMetric, normalize_metric
from exp.framework.results import MetricSeries, NativeResult
from exp.framework.state import StateCoordinate, StateOwner, WorkspacePaths
from exp.parsing import parse_overrides
from mlprosection_mlflow.schema_v1 import build_tags
from mlprosection_mlflow.runtime import RunIdentity


runner = CliRunner()


def test_deepscratch_is_the_canonical_domain_plugin() -> None:
    assert PLUGIN_REGISTRY.names() == ("deepscratch",)
    assert legacy_namespace(Volume.DS2, Variant.ORIGINAL) == "ds2_original"
    coordinate = DeepScratchCoordinate(
        Volume.DS2, "e05", "BETTER-RNNLM", Variant.ORIGINAL
    )
    assert coordinate.atomic_run_id == "BETTER-RNNLM"
    assert coordinate.mlflow_experiment == "deepscratch.ds2"
    assert coordinate.metadata(
        schema_name="ds2-original", schema_version=1, protocol_version="book-v1"
    )["condition.id"] == "BETTER-RNNLM"


def test_workspace_paths_are_owned_and_typed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXP_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("EXP_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    paths = WorkspacePaths.from_environment(tmp_path / "repository")
    coordinate = StateCoordinate(
        "deepscratch", "ds2", "e02", "implemented", "profile"
    )
    assert paths.resolve(StateOwner.CACHE, coordinate) == (
        tmp_path / "cache/deepscratch/ds2/e02/implemented/profile"
    )
    assert paths.resolve(StateOwner.ARTIFACT, coordinate) == (
        tmp_path / "artifacts/deepscratch/ds2/e02/implemented/profile"
    )


def test_new_cli_defaults_to_implemented_and_supports_original_alias() -> None:
    implemented = runner.invoke(
        app, ["plan", "deepscratch", "ds1", "-e", "01", "--seed", "1"]
    )
    original = runner.invoke(
        app, ["plan", "deepscratch", "ds1", "-e", "01", "--seed", "1", "-o"]
    )
    assert implemented.exit_code == 0
    assert "4 planned runs" in implemented.output
    assert original.exit_code == 0
    assert "4 planned runs" in original.output


def test_analyze_original_alias_dispatches_original_variant(monkeypatch) -> None:
    captured = {}

    def fake_analyze_command(definition, **kwargs):
        captured["domain"] = definition.name
        captured.update(kwargs)

    monkeypatch.setattr(
        "exp.deepscratch.cli.analyze_command", fake_analyze_command
    )

    result = runner.invoke(
        app, ["analyze", "deepscratch", "ds2", "-e", "03", "-o", "-s"]
    )

    assert result.exit_code == 0
    assert captured["domain"] == "ds2_original"
    assert captured["experiments"] == ["03"]
    assert captured["summary"] is True


def test_analyze_original_alias_rejects_explicit_variant() -> None:
    result = runner.invoke(
        app,
        [
            "analyze",
            "deepscratch",
            "ds2",
            "-e",
            "03",
            "-o",
            "--variant",
            "all",
        ],
    )

    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_declared_tracking_tags_expand_logical_identity(monkeypatch) -> None:
    class Backend:
        name = "numpy"
        is_gpu = False

    monkeypatch.setattr(
        "mlprosection_mlflow.schema_v1.get_default_backend", lambda: Backend()
    )
    identity = RunIdentity(
        schema_version=1,
        project_name="mlprosection",
        experiment_ids=("e01",),
        atomic_run_id="CONDITION",
        execution_group_id="GT01",
        recipe_id="recipe",
        structure_signature="shape",
        condition_key="condition-key",
        run_key="run-key",
        master_seed=1,
    )
    tags = build_tags(
        identity,
        {
            "protocol_version": "p1",
            "dataset": {},
            "model": {},
            "tracking": {
                "tags": {
                    "domain.name": "deepscratch",
                    "experiment.id": "{experiment_id}",
                    "condition.id": "{condition_id}",
                    "implementation.variant": "implemented",
                }
            },
        },
        {
            "commit": "commit",
            "branch": "branch",
            "dirty": False,
            "repository": "repository",
            "entrypoint": "entrypoint",
        },
        None,
    )
    assert tags["experiment.id"] == "e01"
    assert tags["condition.id"] == "CONDITION"
    assert tags["implementation.variant"] == "implemented"


def test_writer_overrides_preserve_dotted_tag_names_and_templates() -> None:
    overrides = parse_overrides(
        _writer_overrides(Volume.DS2, Variant.IMPLEMENTED, [])
    )

    assert overrides["tracking"] == {
        "experiment": "deepscratch.ds2",
        "tags": {
            "domain.name": "deepscratch",
            "deepscratch.volume": "ds2",
            "implementation.variant": "implemented",
            "experiment.id": "{experiment_id}",
            "condition.id": "{condition_id}",
            "result.schema.name": "ds2-implemented",
            "result.schema.version": "1",
        },
    }


def test_comparison_view_never_synthesizes_missing_metrics() -> None:
    coordinate = DeepScratchCoordinate(
        Volume.DS2, "e03", "RNNLM", Variant.ORIGINAL
    )
    result = NativeResult(
        run_id="run",
        schema_name="ds2-original",
        schema_version=1,
        protocol_version="book-v1",
        metrics=(
            MetricSeries(
                "test/perplexity", "perplexity", "test", "epoch", (1,), (123.0,)
            ),
        ),
    )
    declaration = ComparableMetric(
        metric_id="training_time",
        unit="seconds",
        split="train",
        axis="run",
        evaluation_protocol="book-v1",
        native_metric_id="runtime/train_total_s",
    )
    observation = normalize_metric(coordinate, result, declaration)
    assert observation.available is False
    assert observation.values == ()
    assert observation.unavailable_reason == (
        "native metric is absent: runtime/train_total_s"
    )


def test_variant_and_volume_import_boundaries() -> None:
    root = Path("exp/deepscratch")
    violations = []
    for path in root.rglob("*.py"):
        if "source" in path.parts or "tests" in path.parts:
            continue
        imports = _imports(path)
        relative = path.relative_to(root).parts
        if relative[0] == "ds1":
            violations.extend(
                (path, name) for name in imports
                if name.startswith("exp.deepscratch.ds2")
            )
        if relative[0] == "ds2":
            violations.extend(
                (path, name) for name in imports
                if name.startswith("exp.deepscratch.ds1")
            )
        if "implemented" in relative:
            violations.extend(
                (path, name) for name in imports if ".original" in name
            )
        if "original" in relative:
            violations.extend(
                (path, name) for name in imports if ".implemented" in name
            )
        if relative[:3] == ("ds2", "profile", "e02"):
            violations.extend(
                (path, name) for name in imports
                if ".original.run" in name
            )
    assert violations == []


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names
