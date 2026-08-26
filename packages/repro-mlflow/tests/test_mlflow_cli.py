import json
from pathlib import Path

from typer.testing import CliRunner

import repro_mlflow.mlflow_cli as cli

runner = CliRunner()


def test_export_uses_experiment_name_for_default_archive(
    monkeypatch,
) -> None:
    calls = []

    def fake_export(tracking_uri, experiment, output):
        calls.append((tracking_uri, experiment, output))
        return Path(output)

    monkeypatch.setattr(cli, "export_experiment", fake_export)
    result = runner.invoke(
        cli.app,
        ["export", "ds2", "--tracking-uri", "http://source:5000"],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            "http://source:5000",
            "ds2",
            Path("infra/mlflow/exports/ds2.zip"),
        )
    ]
    assert json.loads(result.stdout)["archive"].endswith("infra/mlflow/exports/ds2.zip")


def test_import_reuses_named_experiment_by_default(monkeypatch) -> None:
    calls = []

    def fake_import(tracking_uri, archive, **kwargs):
        calls.append((tracking_uri, archive, kwargs))
        return {"experiment_id": "7", "reused_experiment": True}

    monkeypatch.setattr(cli, "import_archive", fake_import)
    result = runner.invoke(
        cli.app,
        [
            "import",
            "ds2",
            "--tracking-uri",
            "http://target:5000",
            "--destination-tag",
            "server.name=gpu-a",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            "http://target:5000",
            Path("infra/mlflow/exports/ds2.zip"),
            {
                "experiment_name": "ds2",
                "artifact_location": None,
                "capture_environment": True,
                "destination_tags": {"server.name": "gpu-a"},
                "reuse_experiment": True,
            },
        )
    ]


def test_logs_follow_passes_compose_follow_flag(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_compose", lambda *arguments: calls.append(arguments))

    result = runner.invoke(cli.app, ["logs", "-f", "--tail", "25"])

    assert result.exit_code == 0
    assert calls == [("logs", "--tail", "25", "--follow", "mlflow")]
