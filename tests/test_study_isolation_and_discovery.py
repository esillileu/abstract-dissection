from __future__ import annotations

import types
import warnings
from unittest.mock import MagicMock

import pytest

from repro_core.cli.main import discover_and_register_plugins
from repro_core.execution.runner import run_config


def test_executor_isolation_by_module(monkeypatch) -> None:
    """Verify independent studies can define the same experiment kind without collision."""
    study_a_module = types.ModuleType("study_a.executor")
    study_b_module = types.ModuleType("study_b.executor")

    class StudyAWord2VecExecutor:
        def run(self, config, context):
            return "study_a_word2vec_result"

    class StudyBWord2VecExecutor:
        def run(self, config, context):
            return "study_b_word2vec_result"

    study_a_module.get_executor = lambda kind: (
        StudyAWord2VecExecutor() if kind == "word2vec" else None
    )
    study_b_module.get_executor = lambda kind: (
        StudyBWord2VecExecutor() if kind == "word2vec" else None
    )

    monkeypatch.setitem(__import__("sys").modules, "study_a.executor", study_a_module)
    monkeypatch.setitem(__import__("sys").modules, "study_b.executor", study_b_module)

    result_a = run_config(
        {"kind": "word2vec"},
        executor_module="study_a.executor",
    )
    result_b = run_config(
        {"kind": "word2vec"},
        executor_module="study_b.executor",
    )

    assert result_a == "study_a_word2vec_result"
    assert result_b == "study_b_word2vec_result"


def test_run_config_requires_executor_module() -> None:
    """run_config must raise ValueError if no executor_module is provided."""
    with pytest.raises(ValueError, match="run_config requires an executor_module"):
        run_config({"kind": "word2vec"})


def test_plugin_discovery_warns_on_broken_entrypoint(monkeypatch) -> None:
    """Failing entrypoints should emit visible warnings without breaking healthy ones."""
    broken_ep = MagicMock()
    broken_ep.name = "broken_study"
    broken_ep.value = "broken.plugin:PLUGIN"
    broken_ep.load.side_effect = ImportError("No module named 'broken'")

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group: [broken_ep] if group == "repro.studies" else [],
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        discover_and_register_plugins()

    assert any(
        "Failed to load study plugin 'broken_study'" in str(w.message) for w in recorded
    )
