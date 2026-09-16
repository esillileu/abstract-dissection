"""F2 object-store configuration and restricted-source policy."""

from __future__ import annotations

import os

from repro_io.s3 import S3Config


def s3_config_from_environment(*, restricted: bool = False) -> S3Config:
    prefix = "F2_GIGAWORD_S3" if restricted else "F2_CORPUS_S3"
    values = {
        "endpoint": os.getenv(f"{prefix}_ENDPOINT", ""),
        "root_uri": os.getenv(f"{prefix}_ROOT", ""),
        "access_key": os.getenv(f"{prefix}_ACCESS_KEY", ""),
        "secret_key": os.getenv(f"{prefix}_SECRET_KEY", ""),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise ValueError(f"missing {prefix} configuration: {', '.join(missing)}")
    if not values["root_uri"].startswith("s3://"):
        raise ValueError(f"{prefix}_ROOT must be an s3:// URI")
    return S3Config(**values, region=os.getenv(f"{prefix}_REGION", "us-east-1"))
