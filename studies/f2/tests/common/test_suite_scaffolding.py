"""Tests for F2 sub-study scaffolding, SuiteCatalog, F2Definition registry, and target metrics."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from f2.catalog.cli import app as catalog_cli_app
from f2.cli import app as f2_root_app
from f2.common.analysis.orchestrator import NormalizedMetricSummary
from f2.common.analysis.targets import (
    PaperTargetMetric,
    TargetComparisonResult,
    compare_all_targets,
    compare_to_target,
)
from f2.definition import DEFINITION, F2Definition
from f2.suites.protocol import CHECKPOINT_SOURCE_RESOLVER, SuiteCatalog
from repro_core.execution.definition import ExecutionDefinition

runner = CliRunner()


def test_suite_catalog_build_definition(tmp_path: Path) -> None:
    catalog = SuiteCatalog(
        suite_name="w2v_pretrain",
        root_path=tmp_path,
        description="Word2Vec pretraining suite",
        variant="implemented",
    )
    defn = catalog.build_definition()

    assert defn.name == "f2.w2v_pretrain.implemented"
    assert defn.domain == "f2"
    assert defn.suite == "w2v_pretrain"
    assert defn.variant == "implemented"
    assert defn.config_root == tmp_path / "config"
    assert defn.spec_module == "f2.suites.w2v_pretrain.spec"
    assert defn.executor_module == "f2.suites.w2v_pretrain.executor"
    assert defn.checkpoint_source_resolver_module == CHECKPOINT_SOURCE_RESOLVER


def test_f2_definition_registry(tmp_path: Path) -> None:
    domain = F2Definition()
    assert domain.name == "f2"
    assert domain.suite_names() == ()

    mock_defn = ExecutionDefinition(
        name="f2.benchmarks.implemented",
        config_root=tmp_path / "config",
        spec_module="f2.suites.benchmarks.spec",
        executor_module="f2.suites.benchmarks.executor",
        domain="f2",
        suite="benchmarks",
        variant="implemented",
    )

    domain.register_suite("benchmarks", mock_defn)
    assert domain.has_suite("benchmarks") is True
    assert domain.has_suite("unknown") is False
    assert domain.suite_names() == ("benchmarks",)
    assert domain.get_suite("benchmarks") == mock_defn

    with pytest.raises(ValueError, match="Unknown F2 suite 'nonexistent'"):
        domain.get_suite("nonexistent")


def test_paper_target_comparison() -> None:
    target = PaperTargetMetric(
        metric_id="semantic_accuracy",
        paper_value=0.55,
        description="Semantic analogy accuracy",
        paper_ref="Mikolov et al. (2013) Table 1",
    )

    summary_match = NormalizedMetricSummary(
        metric_id="semantic_accuracy",
        condition_id="dim_300",
        mean_value=0.56,
        std_value=0.01,
        ci_lower=0.54,
        ci_upper=0.58,
        sample_size=5,
    )

    result = compare_to_target(target, summary_match)
    assert isinstance(result, TargetComparisonResult)
    assert result.metric_id == "semantic_accuracy"
    assert result.paper_value == 0.55
    assert result.observed_mean == 0.56
    assert abs(result.absolute_error - 0.01) < 1e-6
    assert abs(result.relative_error - (0.01 / 0.55)) < 1e-6
    assert result.ci_contains_target is True
    assert result.paper_ref == "Mikolov et al. (2013) Table 1"

    summary_miss = NormalizedMetricSummary(
        metric_id="semantic_accuracy",
        condition_id="dim_300",
        mean_value=0.40,
        std_value=0.01,
        ci_lower=0.38,
        ci_upper=0.42,
        sample_size=5,
    )
    result_miss = compare_to_target(target, summary_miss)
    assert result_miss.ci_contains_target is False

    batch = compare_all_targets([target], [summary_match])
    assert len(batch) == 1
    assert batch[0].metric_id == "semantic_accuracy"


def test_f2_cli_suites_command(tmp_path: Path) -> None:
    result = runner.invoke(f2_root_app, ["suites"])
    assert result.exit_code == 0
    assert "suites" in result.output.lower()

    # Test with registered suite
    mock_defn = ExecutionDefinition(
        name="f2.dummy_suite.implemented",
        config_root=tmp_path / "config",
        spec_module="f2.suites.dummy.spec",
        executor_module="f2.suites.dummy.executor",
        domain="f2",
        suite="dummy_suite",
        variant="implemented",
    )
    DEFINITION.register_suite("dummy_suite", mock_defn)
    try:
        result_with_suite = runner.invoke(f2_root_app, ["suites"])
        assert result_with_suite.exit_code == 0
        assert "dummy_suite" in result_with_suite.output
    finally:
        # cleanup
        DEFINITION._suites.pop("dummy_suite", None)


def test_f2_catalog_materialize_missing_suite() -> None:
    result = runner.invoke(catalog_cli_app, ["materialize", "non_existent_suite"])
    assert result.exit_code != 0
    assert "Unknown F2 suite" in result.output or "Error" in result.output
