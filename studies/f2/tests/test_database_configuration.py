"""Freeze the existing dotenv order while removing implicit credentials."""

import pytest

from f2.catalog.db.session import CatalogDatabaseConfig, CatalogDatabaseConfigError
from f2.corpus.db.session import DatabaseConfig, DatabaseConfigError


@pytest.mark.parametrize(
    "config,error,alias",
    [
        (CatalogDatabaseConfig, CatalogDatabaseConfigError, "F2_CATALOG_DATABASE_URL"),
        (DatabaseConfig, DatabaseConfigError, "F2_CORPUS_DATABASE_URL"),
    ],
)
def test_database_resolution(config, error, alias, monkeypatch):
    import dotenv

    for key in ("F2_DATABASE_URL", "F2_CATALOG_DATABASE_URL", "F2_CORPUS_DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(error, match="required"):
        config.from_environment()
    assert calls == [{"override": True}]
    monkeypatch.setenv(alias, "postgresql://legacy/example")
    assert config.from_environment().connection_url == "postgresql://legacy/example"

    def load(**kwargs):
        assert kwargs == {"override": True}
        monkeypatch.setenv("F2_DATABASE_URL", "postgresql://dotenv/example")

    monkeypatch.setattr(dotenv, "load_dotenv", load)
    assert config.from_environment().connection_url == "postgresql://dotenv/example"
    monkeypatch.setenv("F2_DATABASE_URL", "postgresql://explicit-env/example")
    assert (
        config.from_environment().connection_url == "postgresql://explicit-env/example"
    )


@pytest.mark.parametrize("module", ["f2.catalog.db.session", "f2.corpus.db.session"])
def test_explicit_connection_url_bypasses_environment(module, monkeypatch):
    import importlib
    from unittest.mock import MagicMock

    session = importlib.import_module(module)
    connect = MagicMock()
    monkeypatch.setattr(session.psycopg, "connect", connect)
    with session.get_connection("postgresql://explicit/example"):
        pass
    assert connect.call_args.args == ("postgresql://explicit/example",)


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
