"""Freeze the existing dotenv order while removing implicit credentials."""

import pytest

from f2.corpus.db.session import DatabaseConfig, DatabaseConfigError


def test_database_resolution(monkeypatch):
    import dotenv

    for key in ("F2_DATABASE_URL", "F2_CORPUS_DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(DatabaseConfigError, match="required"):
        DatabaseConfig.from_environment()
    assert calls == [{"override": True}]
    monkeypatch.setenv("F2_CORPUS_DATABASE_URL", "postgresql://legacy/example")
    with pytest.raises(DatabaseConfigError, match="F2_DATABASE_URL is required"):
        DatabaseConfig.from_environment()

    def load(**kwargs):
        assert kwargs == {"override": True}
        monkeypatch.setenv("F2_DATABASE_URL", "postgresql://dotenv/f2")

    monkeypatch.setattr(dotenv, "load_dotenv", load)
    assert DatabaseConfig.from_environment().connection_url == "postgresql://dotenv/f2"
    monkeypatch.setenv("F2_DATABASE_URL", "postgresql://explicit-env/f2")
    assert (
        DatabaseConfig.from_environment().connection_url
        == "postgresql://explicit-env/f2"
    )

    monkeypatch.setenv("F2_DATABASE_URL", "postgresql://explicit-env/f2_db")
    with pytest.raises(DatabaseConfigError, match="canonical 'f2'"):
        DatabaseConfig.from_environment()


def test_explicit_connection_url_bypasses_environment(monkeypatch):
    import importlib
    from unittest.mock import MagicMock

    session = importlib.import_module("f2.corpus.db.session")
    connect = MagicMock()
    monkeypatch.setattr(session.psycopg, "connect", connect)
    with session.get_connection("postgresql://explicit/f2_test"):
        pass
    assert connect.call_args.args == ("postgresql://explicit/f2_test",)


@pytest.mark.parametrize(
    "restricted,prefix", [(False, "F2_CORPUS_S3"), (True, "F2_GIGAWORD_S3")]
)
def test_study_s3_configuration_is_explicit_and_isolated(
    restricted, prefix, monkeypatch
):
    from f2.corpus.object_store import s3_config_from_environment

    for suffix in ("ENDPOINT", "ROOT", "ACCESS_KEY", "SECRET_KEY", "REGION"):
        monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    with pytest.raises(ValueError, match=f"missing {prefix}"):
        s3_config_from_environment(restricted=restricted)
    for suffix, value in {
        "ENDPOINT": "https://example.test",
        "ROOT": "s3://fixture-bucket/prefix",
        "ACCESS_KEY": "fixture-access",
        "SECRET_KEY": "fixture-secret",
    }.items():
        monkeypatch.setenv(f"{prefix}_{suffix}", value)
    config = s3_config_from_environment(restricted=restricted)
    assert (config.root_uri, config.region, config.access_key) == (
        "s3://fixture-bucket/prefix",
        "us-east-1",
        "fixture-access",
    )
    monkeypatch.setenv(f"{prefix}_ROOT", "https://wrong.test")
    with pytest.raises(ValueError, match="s3://"):
        s3_config_from_environment(restricted=restricted)
