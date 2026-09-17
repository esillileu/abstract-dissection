from __future__ import annotations

import pytest

from f2.preflight import PreflightError, corpus_store_identity, mlflow_identity


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
@pytest.mark.parametrize("port", [9000, 19000])
def test_corpus_store_preflight_accepts_local_ports(monkeypatch, host, port):
    from f2 import preflight

    rendered_host = f"[{host}]" if host == "::1" else host
    config = type(
        "Config",
        (),
        {
            "endpoint": f"http://{rendered_host}:{port}",
            "root_uri": "s3://bucket/corpus",
        },
    )()
    monkeypatch.setattr(preflight, "s3_config_from_environment", lambda: config)
    assert corpus_store_identity(probe=False).host == host


@pytest.mark.parametrize(
    "endpoint",
    ["http://127.0.0.1:9001", "http://s3.example.test:9000", "https://localhost:9000"],
)
def test_corpus_store_preflight_rejects_other_cleartext_targets(monkeypatch, endpoint):
    from f2 import preflight

    config = type(
        "Config",
        (),
        {"endpoint": endpoint, "root_uri": "s3://bucket/corpus"},
    )()
    monkeypatch.setattr(preflight, "s3_config_from_environment", lambda: config)
    with pytest.raises(PreflightError, match="local HTTP"):
        corpus_store_identity(probe=False)


def test_preflight_identities_never_include_credentials(monkeypatch):
    from f2 import preflight

    config = type(
        "Config",
        (),
        {
            "endpoint": "https://access:secret@worker.tailnet.ts.net:9000",
            "root_uri": "s3://fixture/corpus",
        },
    )()
    monkeypatch.setattr(preflight, "s3_config_from_environment", lambda: config)
    corpus = corpus_store_identity(probe=False)
    mlflow = mlflow_identity("https://token@mlflow.example.test/path", probe=False)
    rendered = repr((corpus, mlflow))
    assert corpus.host == "worker.tailnet.ts.net"
    assert mlflow.host == "mlflow.example.test"
    assert "access" not in rendered
    assert "secret" not in rendered
    assert "token" not in rendered


def test_mlflow_preflight_rejects_database_uri():
    with pytest.raises(PreflightError, match=r"HTTP\(S\)"):
        mlflow_identity("postgresql://db.example/f2", probe=False)
