# MLflow Infrastructure Archives

This directory serves as the repository archive for historical migration scripts, transition utilities, one-off schema transformation tools, and verification reports related to the MLflow tracking infrastructure.

## Contents

* **`20260828_f1_sqlite_to_postgres_migration.py`**:
  * Standalone migration script used to transfer the full DLFS experiment dataset (SQLite `mlflow.db` and ~26GB file artifacts) to the PostgreSQL-backed MLflow service (`mlflow_f1`, port 5001).
  * Implements per-table atomic commits, PostgreSQL streaming `COPY` for ~30M metric rows, and 24-worker parallel HTTP artifact uploads.
* **`20260828_f1_sqlite_to_postgres_migration_summary.json`**:
  * Integrity verification report and record of row counts across all 16 MLflow tables and 1,598 runs migrated on 2026-08-28.

## Usage & Guidelines

* Files in this directory are preserved for reference, auditability, and reusable templates for future database/service migrations.
* New migration scripts or transition artifacts should follow the naming convention `<YYYYMMDD>_<purpose>.<ext>`.
