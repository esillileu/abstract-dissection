"""Validation profile and test assertion execution history operations."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCorpusRepository


class ValidationRepositoryMixin(BaseCorpusRepository):
    """Repository operations for validation profiles, validation runs, and assertion checks."""

    def create_validation_profile(
        self,
        profile_id: str,
        profile_key: str,
        revision: int,
        name: str,
        spec_hash: str,
        specification: dict[str, Any],
        description: str | None = None,
        target_paper_id: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Create an immutable validation profile specification."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO validation_profiles (
                    profile_id, profile_key, revision, name, description,
                    target_paper_id, spec_hash, specification, is_active, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s);
                """,
                (
                    profile_id,
                    profile_key,
                    revision,
                    name,
                    description,
                    target_paper_id,
                    spec_hash,
                    json.dumps(specification),
                    notes,
                ),
            )
        self._commit()
        return profile_id

    def deprecate_validation_profile(
        self, profile_id: str, notes: str | None = None
    ) -> None:
        """Deprecate a validation profile revision (without modifying specification)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE validation_profiles
                SET is_active = FALSE,
                    superseded_at = NOW(),
                    notes = COALESCE(%s, notes)
                WHERE profile_id = %s;
                """,
                (notes, profile_id),
            )
        self._commit()

    def get_validation_profile(self, profile_id: str) -> dict[str, Any] | None:
        """Retrieve validation profile by ID."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM validation_profiles WHERE profile_id = %s;",
                (profile_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))

    def record_validation_run(
        self,
        validation_run_id: str,
        profile_id: str,
        target_type: str,
        target_id: str,
        validator_code_version: str,
        validator_config_hash: str,
        overall_verdict: str,
        checks: list[dict[str, Any]],
        paper_compatibility: str = "not_applicable",
        validator_config: dict[str, Any] | None = None,
        evidence_report_uri: str | None = None,
        summary_metrics: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> str:
        """Record validation results for a processing run, artifact, or resource version."""
        target_proc_id = target_id if target_type == "processing_run" else None
        target_art_id = target_id if target_type == "artifact" else None
        target_rv_id = target_id if target_type == "resource_version" else None

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO validation_runs (
                    validation_run_id, profile_id, target_type,
                    target_processing_run_id, target_artifact_id, target_resource_version_id,
                    validator_code_version, validator_config_hash, validator_config,
                    overall_verdict, paper_compatibility, evidence_report_uri,
                    summary_metrics, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    validation_run_id,
                    profile_id,
                    target_type,
                    target_proc_id,
                    target_art_id,
                    target_rv_id,
                    validator_code_version,
                    validator_config_hash,
                    json.dumps(validator_config or {}),
                    overall_verdict,
                    paper_compatibility,
                    evidence_report_uri,
                    json.dumps(summary_metrics or {}),
                    notes,
                ),
            )
            for check in checks:
                cur.execute(
                    """
                    INSERT INTO validation_checks (
                        validation_run_id, check_name, category, status,
                        expected_condition, observed_value, details
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        validation_run_id,
                        check["check_name"],
                        check.get("category", "integrity"),
                        check["status"],
                        str(check.get("expected_condition", "")),
                        str(check.get("observed_value", "")),
                        json.dumps(check.get("details", {})),
                    ),
                )
        self._commit()
        return validation_run_id

    def get_validation_run(self, validation_run_id: str) -> dict[str, Any] | None:
        """Retrieve validation run along with its assertion checks."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM validation_runs WHERE validation_run_id = %s;",
                (validation_run_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            res = dict(zip(cols, row, strict=False))

            cur.execute(
                "SELECT * FROM validation_checks WHERE validation_run_id = %s ORDER BY check_id;",
                (validation_run_id,),
            )
            check_cols = [desc[0] for desc in cur.description]
            res["checks"] = [
                dict(zip(check_cols, c_row, strict=False)) for c_row in cur.fetchall()
            ]
            return res

    def get_validation_history(
        self, target_type: str, target_id: str
    ) -> list[dict[str, Any]]:
        """Retrieve validation history for a specific target ordered by execution time."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT vr.*, vp.profile_key, vp.revision AS profile_revision, vp.spec_hash
                FROM validation_runs vr
                JOIN validation_profiles vp ON vp.profile_id = vr.profile_id
                WHERE vr.target_type = %s AND vr.target_id = %s
                ORDER BY vr.started_at DESC;
                """,
                (target_type, target_id),
            )
            cols = [desc[0] for desc in cur.description]
            runs = [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
            for run in runs:
                cur.execute(
                    "SELECT * FROM validation_checks WHERE validation_run_id = %s ORDER BY check_id;",
                    (run["validation_run_id"],),
                )
                c_cols = [desc[0] for desc in cur.description]
                run["checks"] = [
                    dict(zip(c_cols, c_row, strict=False)) for c_row in cur.fetchall()
                ]
            return runs


__all__ = ["ValidationRepositoryMixin"]
