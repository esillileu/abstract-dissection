"""Artifact validation and integrity checks prior to MLflow publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from f2.execution.identity import ConditionIdentity


def validate_local_artifacts(
    run_root: Path,
    condition: ConditionIdentity,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate result.json, complete=True, strict identity match, artifact paths, and manifests.

    Raises ValueError if local artifacts are missing, corrupt, incomplete, or invalid.
    Returns (result_dict, artifacts_dict).
    """
    result_file = run_root / "result.json"
    if not result_file.is_file():
        raise ValueError(f"Cannot publish run: missing result.json in {run_root}")

    try:
        res_data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot publish run: corrupted result.json: {exc}") from exc

    if not res_data.get("complete"):
        raise ValueError(
            f"Cannot publish incomplete run: complete={res_data.get('complete')}"
        )

    res_identity = res_data.get("identity")
    if not isinstance(res_identity, dict):
        raise ValueError(
            "Cannot publish run: missing identity dictionary in result.json"
        )

    res_digest = res_identity.get("config_digest")
    if not res_digest or res_digest != condition.config_digest:
        raise ValueError(
            f"Cannot publish run: config_digest mismatch ({res_digest} != {condition.config_digest})"
        )
    res_seed = res_identity.get("seed")
    if res_seed is None or int(res_seed) != condition.seed:
        raise ValueError(
            f"Cannot publish run: seed mismatch ({res_seed} != {condition.seed})"
        )
    res_slot = res_identity.get("planned_run_slot_id")
    if not res_slot or (
        condition.planned_run_slot_id and res_slot != condition.planned_run_slot_id
    ):
        raise ValueError(
            f"Cannot publish run: slot_id mismatch ({res_slot} != {condition.planned_run_slot_id})"
        )

    artifacts = res_data.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(
            "Cannot publish run: missing artifacts dictionary in result.json"
        )

    lookup_val = artifacts.get("lookup")
    if not lookup_val or not isinstance(lookup_val, str):
        raise ValueError("Cannot publish run: missing lookup in artifacts")
    checkpoint_val = artifacts.get("checkpoint")
    if not checkpoint_val or not isinstance(checkpoint_val, str):
        raise ValueError("Cannot publish run: missing checkpoint in artifacts")
    metrics_val = artifacts.get("metrics")
    if not metrics_val or not isinstance(metrics_val, str):
        raise ValueError("Cannot publish run: missing metrics in artifacts")

    final_lookup_path = (run_root / lookup_val).resolve()
    if not final_lookup_path.is_relative_to(run_root.resolve()):
        raise ValueError(
            f"Cannot publish run: lookup path escapes run directory: {lookup_val}"
        )
    if not final_lookup_path.exists():
        raise ValueError(
            f"Cannot publish run: designated lookup does not exist: {final_lookup_path}"
        )
    if not final_lookup_path.is_dir():
        raise ValueError(
            f"Cannot publish run: designated lookup is not a directory: {final_lookup_path}"
        )

    chk_path = (run_root / checkpoint_val).resolve()
    if not chk_path.is_relative_to(run_root.resolve()):
        raise ValueError(
            f"Cannot publish run: checkpoint path escapes run directory: {checkpoint_val}"
        )
    if not chk_path.exists():
        raise ValueError(
            f"Cannot publish run: designated checkpoint does not exist: {chk_path}"
        )
    if not chk_path.is_dir():
        raise ValueError(
            f"Cannot publish run: designated checkpoint is not a directory: {chk_path}"
        )

    for art_name, art_path_rel in [
        ("lookup", final_lookup_path),
        ("checkpoint", chk_path),
    ]:
        if art_path_rel.is_dir():
            manifest_f = art_path_rel / "manifest.json"
            if not manifest_f.is_file():
                raise ValueError(
                    f"Cannot publish run: designated {art_name} directory lacks manifest.json: {art_path_rel}"
                )
            try:
                fmt = json.loads(manifest_f.read_text(encoding="utf-8")).get("format")
            except Exception as exc:
                raise ValueError(
                    f"Cannot publish run: failed to parse {art_name} manifest in {art_path_rel}: {exc}"
                ) from exc
            if not fmt or "unknown" in fmt:
                raise ValueError(
                    f"Cannot publish run: unsupported {art_name} format '{fmt}' in {art_path_rel}"
                )

    met_path = (run_root / metrics_val).resolve()
    if not met_path.is_relative_to(run_root.resolve()):
        raise ValueError(
            f"Cannot publish run: metrics path escapes run directory: {metrics_val}"
        )
    if not met_path.exists():
        raise ValueError(
            f"Cannot publish run: designated metrics file does not exist: {met_path}"
        )

    return res_data, artifacts
