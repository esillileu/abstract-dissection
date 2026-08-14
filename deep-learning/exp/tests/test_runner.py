from pathlib import Path
import warnings

import pytest

from exp.framework.execution import runner


def test_remove_durable_staging_ignores_already_removed_root(monkeypatch, tmp_path: Path) -> None:
    def missing(_path: Path) -> None:
        raise FileNotFoundError(2, "No such file or directory", str(tmp_path))

    monkeypatch.setattr(runner.shutil, "rmtree", missing)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        runner._remove_durable_staging(tmp_path / "staging")

    assert not recorded


def test_remove_durable_staging_warns_for_other_cleanup_errors(monkeypatch, tmp_path: Path) -> None:
    def denied(_path: Path) -> None:
        raise PermissionError(13, "Permission denied", str(tmp_path))

    monkeypatch.setattr(runner.shutil, "rmtree", denied)

    with pytest.warns(UserWarning, match="verified staging was preserved"):
        runner._remove_durable_staging(tmp_path / "staging")
