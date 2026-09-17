"""Constants and serialization shared by the 2026 integration commands."""

from __future__ import annotations

import hashlib
import json
from typing import Any

EXPECTED_CLEANUP_RUNS = 430
CANONICAL_PROFILES = {
    "calibration-10k": "run_42_a1d3745e",
    "confirmatory-50k": "run_50k_confirmatory",
}
LATEST_MIGRATION = "002_release_lineage"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def manifest_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
