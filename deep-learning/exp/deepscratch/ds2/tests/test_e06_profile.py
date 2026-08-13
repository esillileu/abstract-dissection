from __future__ import annotations

from typer.testing import CliRunner

from exp.cli import app


def test_e06_profile_cli_dispatches_to_isolated_profiler(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        "exp.deepscratch.ds2.profile.e06.benchmark.run",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        [
            "profile", "deepscratch", "ds2", "-e", "06", "--device", "cpu",
            "--condition", "vanilla-forward", "--update-warmup", "3",
            "--measured-updates", "7", "--update-repetitions", "2",
            "--output-dir", str(tmp_path),
            "--no-record",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "device": "cpu",
        "conditions": ("vanilla-forward",),
        "warmup": 3,
        "iterations": 7,
        "repetitions": 2,
        "output_dir": tmp_path,
    }
