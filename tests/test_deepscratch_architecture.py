import ast
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from dlfs.analysis.normalization import ComparableMetric, normalize_metric
from dlfs.cli import _writer_overrides
from dlfs.identity import DeepScratchCoordinate, Variant, Volume
from repro_core.cli import PLUGIN_REGISTRY, app
from repro_core.context.paths import StateCoordinate, StateOwner, WorkspacePaths
from repro_core.execution.parsing import parse_overrides
from repro_core.results import MetricSeries, NativeRunResult
from repro_mlflow.runtime import RunIdentity
from repro_mlflow.schema_v1 import build_tags

runner = CliRunner()


def test_deepscratch_is_the_canonical_domain_plugin() -> None:
    assert "dlfs" in PLUGIN_REGISTRY.names() or "deepscratch" in PLUGIN_REGISTRY.names()
    coordinate = DeepScratchCoordinate(
        Volume.DS2, "e05", "BETTER-RNNLM", Variant.ORIGINAL
    )
    assert coordinate.condition_id == "BETTER-RNNLM"
    assert coordinate.mlflow_experiment == "deepscratch.ds2"
    assert (
        coordinate.metadata(
            schema_name="ds2-original", schema_version=1, protocol_version="book-v1"
        )["condition.id"]
        == "BETTER-RNNLM"
    )


def test_workspace_paths_are_owned_and_typed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXP_STAGING_ROOT", str(tmp_path / "staging"))
    monkeypatch.setenv("EXP_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("EXP_RESULTS_ROOT", str(tmp_path / "results"))
    paths = WorkspacePaths.from_environment(tmp_path / "repository")
    coordinate = StateCoordinate("deepscratch", "ds2", "e02", "implemented", "profile")
    assert paths.resolve(StateOwner.CACHE, coordinate) == (
        tmp_path / "cache/exp/deepscratch/ds2/e02/implemented/profile"
    )
    assert paths.resolve(StateOwner.RESULTS, coordinate) == (
        tmp_path / "results/exp/deepscratch"
    )
    assert paths.resolve(StateOwner.STAGING, coordinate) == (
        tmp_path / "staging/exp/deepscratch/ds2/e02/implemented/profile"
    )


def test_workspace_path_defaults_match_storage_lifecycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in (
        "EXP_STAGING_ROOT",
        "EXP_CACHE_ROOT",
        "EXP_RESULTS_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    paths = WorkspacePaths.from_environment(tmp_path)
    assert paths.staging_root == tmp_path / ".staging"
    assert paths.results_root == tmp_path / "artifacts"
    assert paths.cache_root == tmp_path / ".cache"


def test_workspace_paths_reject_coordinate_and_relative_root_escape(
    tmp_path: Path, monkeypatch
) -> None:
    paths = WorkspacePaths.from_environment(tmp_path)
    with pytest.raises(ValueError, match="invalid state coordinate"):
        paths.run_staging(
            domain="deepscratch",
            suite="ds2",
            study="..",
            variant="implemented",
            run_key="run",
        )
    monkeypatch.setenv("EXP_CACHE_ROOT", "../outside")
    with pytest.raises(ValueError, match="escapes repository root"):
        WorkspacePaths.from_environment(tmp_path)


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


def test_analyze_original_alias_dispatches_original_variant(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    def fake_write_analysis(*args, **kwargs):
        captured["tracking_uri"] = args[0]
        captured.update(kwargs)
        return tmp_path / "observations.csv"

    monkeypatch.setattr(
        "dlfs.analysis.orchestrator.write_analysis",
        fake_write_analysis,
    )

    result = runner.invoke(app, ["analyze", "deepscratch", "ds2", "-e", "03", "-o"])

    assert result.exit_code == 0
    assert captured["experiment_ids"] == ["e03"]
    assert captured["variants"] == (Variant.ORIGINAL,)
    assert captured["error_style"] == "band"


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


def test_declared_tracking_tags_expand_logical_identity() -> None:
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
    overrides = parse_overrides(_writer_overrides(Volume.DS2, Variant.IMPLEMENTED, []))

    assert overrides["tracking"] == {
        "experiment": "deepscratch.ds2",
        "tags": {
            "domain.name": "deepscratch",
            "suite.name": "ds2",
            "deepscratch.volume": "ds2",
            "implementation.variant": "implemented",
            "experiment.id": "{experiment_id}",
            "condition.id": "{condition_id}",
            "result.schema.name": "ds2-implemented",
            "result.schema.version": "1",
        },
    }


def test_comparison_view_never_synthesizes_missing_metrics() -> None:
    coordinate = DeepScratchCoordinate(Volume.DS2, "e03", "RNNLM", Variant.ORIGINAL)
    result = NativeRunResult(
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
    root = Path("studies/dlfs/src/dlfs")
    violations = []
    for path in root.rglob("*.py"):
        if "source" in path.parts or "tests" in path.parts:
            continue
        imports = _imports(path)
        relative = path.relative_to(root).parts
        if relative[0] == "ds1":
            violations.extend(
                (path, name) for name in imports if name.startswith("dlfs.ds2")
            )
        if relative[0] == "ds2":
            violations.extend(
                (path, name) for name in imports if name.startswith("dlfs.ds1")
            )
        if "implemented" in relative:
            violations.extend((path, name) for name in imports if ".original" in name)
        if "original" in relative:
            violations.extend(
                (path, name) for name in imports if ".implemented" in name
            )
        if relative[:3] == ("ds2", "profile", "e02"):
            violations.extend(
                (path, name) for name in imports if ".original.run" in name
            )
    assert violations == []


def test_generic_layers_do_not_know_deepscratch() -> None:
    violations = []
    roots = (Path("packages/repro-core/src"), Path("packages/repro-mlflow/src"))
    for root in roots:
        for path in root.rglob("*.py"):
            violations.extend(
                (path, name)
                for name in _imports(path)
                if name.startswith("dlfs") or name.startswith("exp.deepscratch")
            )
    assert violations == []


def test_retired_python_facades_are_absent() -> None:
    assert not Path("deep-learning").exists()
    assert not Path("exp").exists()


def test_monorepo_package_structure() -> None:
    assert Path("packages/repro-core").is_dir()
    assert Path("packages/repro-mlflow").is_dir()
    assert Path("packages/deepscratch").is_dir()
    assert Path("studies/dlfs").is_dir()
    assert Path("references/dlfs1-book").is_dir()
    assert Path("references/dlfs2-book").is_dir()


def test_runtime_code_does_not_write_under_source_tree() -> None:
    violations = []
    forbidden = (
        'ROOT / "results"',
        'Path(".artifacts/experiments',
        'Path(".cache/experiments',
    )
    for path in Path("studies/dlfs").rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden) or re.search(
            r'Path\("studies/dlfs/[^"]*/results', text
        ):
            violations.append(path)
    assert violations == []


def test_suite_declarations_cover_every_catalog_condition() -> None:
    for volume in (Volume.DS1, Volume.DS2):
        schema = __import__(
            f"dlfs.{volume.value}.result_schema",
            fromlist=["STUDIES"],
        )
        for variant in Variant:
            root = Path(f"studies/dlfs/src/dlfs/{volume.value}/config/{variant.value}")
            for path in sorted(root.glob("e[0-9][0-9]_*.yaml")):
                source = yaml.safe_load(path.read_text(encoding="utf-8"))
                if source.get("kind") == "performance_profile":
                    continue
                catalog = set(source["variants"])
                study = schema.STUDIES[path.name[:3]]
                declared = {
                    alias
                    for condition in study.conditions
                    for alias in condition.aliases(variant)
                }
                assert declared == catalog, (volume, variant, path, declared ^ catalog)


def test_current_configs_separate_execution_identity_from_storage_namespace() -> None:
    for volume in (Volume.DS1, Volume.DS2):
        for variant in Variant:
            root = Path(f"studies/dlfs/src/dlfs/{volume.value}/config/{variant.value}")
            for path in root.glob("e*.yaml"):
                source = yaml.safe_load(path.read_text(encoding="utf-8"))
                assert source["domain"] == (
                    f"deepscratch.{volume.value}.{variant.value}"
                )
                assert source["tracking"]["experiment"] == (
                    f"deepscratch.{volume.value}"
                )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_deepscratch_package_has_zero_dependencies_on_repro_core() -> None:
    violations = []
    root = Path("packages/deepscratch/src")
    for path in root.rglob("*.py"):
        violations.extend(
            (path, name)
            for name in _imports(path)
            if name.startswith("repro_core")
            or name.startswith("repro_mlflow")
            or name.startswith("dlfs")
        )
    assert violations == []


def test_repro_core_has_zero_dependencies_on_deepscratch_or_dlfs() -> None:
    violations = []
    root = Path("packages/repro-core/src")
    for path in root.rglob("*.py"):
        violations.extend(
            (path, name)
            for name in _imports(path)
            if name.startswith("deepscratch") or name.startswith("dlfs")
        )
    assert violations == []


def test_repro_core_checkpoint_has_zero_deep_learning_coupling() -> None:
    path = Path("packages/repro-core/src/repro_core/context/checkpoint.py")
    text = path.read_text(encoding="utf-8")
    forbidden = (
        "named_parameters",
        "named_buffers",
        "save_params_npz",
        "load_params_npz",
        "random_stream_states",
        "restore_random_stream_states",
        "import pickle",
        "import numpy",
    )
    for token in forbidden:
        assert token not in text, (
            f"repro_core/context/checkpoint.py contains forbidden token: {token}"
        )


def test_repro_mlflow_has_zero_dependencies_on_deepscratch_or_dlfs() -> None:
    violations = []
    root = Path("packages/repro-mlflow/src")
    for path in root.rglob("*.py"):
        violations.extend(
            (path, name)
            for name in _imports(path)
            if name.startswith("deepscratch") or name.startswith("dlfs")
        )
    assert violations == []
