#!/usr/bin/env python3
"""Migrate SQLite MLflow data and artifacts to PostgreSQL MLflow instance."""

from __future__ import annotations

import concurrent.futures
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import psycopg
from mlflow.tracking import MlflowClient
from tqdm import tqdm

SQLITE_DB_PATH = Path("f1-migrate/infra/mlflow/data/mlflow.db")
ARTIFACTS_ROOT = Path("f1-migrate/infra/mlflow/data/artifacts")
MLFLOW_SERVER_URL = "http://127.0.0.1:5001"


def get_pg_connection() -> psycopg.Connection[Any]:
    db_url = os.environ.get("MLFLOW_F1_DATABASE_URL")
    if not db_url:
        raise RuntimeError("MLFLOW_F1_DATABASE_URL is not set.")
    return psycopg.connect(db_url)


def migrate_database(
    sqlite_conn: sqlite3.Connection, pg_conn: psycopg.Connection[Any]
) -> dict[str, int]:
    """Migrate all metadata tables from SQLite to PostgreSQL in FK-safe order."""
    cur_s = sqlite_conn.cursor()
    counts: dict[str, int] = {}

    print("=== Step 1: Migrating Database Metadata ===")

    # 1. Experiments
    print("1. Migrating experiments...")
    with pg_conn.cursor() as cur_p:
        exp_rows = cur_s.execute(
            "SELECT experiment_id, name, artifact_location, lifecycle_stage, creation_time, last_update_time, workspace FROM experiments"
        ).fetchall()
        for r in exp_rows:
            cur_p.execute(
                """
                INSERT INTO experiments (experiment_id, name, artifact_location, lifecycle_stage, creation_time, last_update_time, workspace)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (experiment_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    artifact_location = EXCLUDED.artifact_location,
                    lifecycle_stage = EXCLUDED.lifecycle_stage,
                    creation_time = EXCLUDED.creation_time,
                    last_update_time = EXCLUDED.last_update_time,
                    workspace = EXCLUDED.workspace
                """,
                r,
            )
        counts["experiments"] = len(exp_rows)
    pg_conn.commit()

    # 2. Experiment Tags
    print("2. Migrating experiment_tags...")
    with pg_conn.cursor() as cur_p:
        exp_tag_rows = cur_s.execute(
            "SELECT key, value, experiment_id FROM experiment_tags"
        ).fetchall()
        for r in exp_tag_rows:
            cur_p.execute(
                """
                INSERT INTO experiment_tags (key, value, experiment_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (key, experiment_id) DO UPDATE SET
                    value = EXCLUDED.value
                """,
                r,
            )
        counts["experiment_tags"] = len(exp_tag_rows)
    pg_conn.commit()

    # 3. Runs
    print("3. Migrating runs...")
    with pg_conn.cursor() as cur_p:
        run_rows = cur_s.execute(
            """SELECT run_uuid, name, source_type, source_name, entry_point_name,
                      user_id, status, start_time, end_time, source_version,
                      lifecycle_stage, artifact_uri, experiment_id, deleted_time
               FROM runs"""
        ).fetchall()
        for r in run_rows:
            cur_p.execute(
                """
                INSERT INTO runs (run_uuid, name, source_type, source_name, entry_point_name,
                                  user_id, status, start_time, end_time, source_version,
                                  lifecycle_stage, artifact_uri, experiment_id, deleted_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_uuid) DO UPDATE SET
                    name = EXCLUDED.name,
                    status = EXCLUDED.status,
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time,
                    lifecycle_stage = EXCLUDED.lifecycle_stage,
                    artifact_uri = EXCLUDED.artifact_uri,
                    experiment_id = EXCLUDED.experiment_id,
                    deleted_time = EXCLUDED.deleted_time
                """,
                r,
            )
        counts["runs"] = len(run_rows)
    pg_conn.commit()

    # 4. Tags
    print("4. Migrating tags...")
    with pg_conn.cursor() as cur_p:
        tag_rows = cur_s.execute("SELECT key, value, run_uuid FROM tags").fetchall()
        for r in tag_rows:
            cur_p.execute(
                """
                INSERT INTO tags (key, value, run_uuid)
                VALUES (%s, %s, %s)
                ON CONFLICT (key, run_uuid) DO UPDATE SET
                    value = EXCLUDED.value
                """,
                r,
            )
        counts["tags"] = len(tag_rows)
    pg_conn.commit()

    # 5. Params
    print("5. Migrating params...")
    with pg_conn.cursor() as cur_p:
        param_rows = cur_s.execute("SELECT key, value, run_uuid FROM params").fetchall()
        for r in param_rows:
            cur_p.execute(
                """
                INSERT INTO params (key, value, run_uuid)
                VALUES (%s, %s, %s)
                ON CONFLICT (key, run_uuid) DO UPDATE SET
                    value = EXCLUDED.value
                """,
                r,
            )
        counts["params"] = len(param_rows)
    pg_conn.commit()

    # 6. Latest Metrics
    print("6. Migrating latest_metrics...")
    with pg_conn.cursor() as cur_p:
        latest_metric_rows = cur_s.execute(
            "SELECT key, value, timestamp, step, is_nan, run_uuid FROM latest_metrics"
        ).fetchall()
        for r in latest_metric_rows:
            cur_p.execute(
                """
                INSERT INTO latest_metrics (key, value, timestamp, step, is_nan, run_uuid)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (key, run_uuid) DO UPDATE SET
                    value = EXCLUDED.value,
                    timestamp = EXCLUDED.timestamp,
                    step = EXCLUDED.step,
                    is_nan = EXCLUDED.is_nan
                """,
                (r[0], r[1], r[2], r[3], bool(r[4]), r[5]),
            )
        counts["latest_metrics"] = len(latest_metric_rows)
    pg_conn.commit()

    # 7. Registered Models & Model Versions
    print("7. Migrating registered_models & model_versions...")
    with pg_conn.cursor() as cur_p:
        reg_model_rows = cur_s.execute(
            "SELECT name, creation_time, last_updated_time, description, workspace FROM registered_models"
        ).fetchall()
        for r in reg_model_rows:
            cur_p.execute(
                """
                INSERT INTO registered_models (name, creation_time, last_updated_time, description, workspace)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (workspace, name) DO NOTHING
                """,
                r,
            )
        counts["registered_models"] = len(reg_model_rows)

        reg_model_tag_rows = cur_s.execute(
            "SELECT key, value, name, workspace FROM registered_model_tags"
        ).fetchall()
        for r in reg_model_tag_rows:
            cur_p.execute(
                """
                INSERT INTO registered_model_tags (key, value, name, workspace)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (workspace, key, name) DO NOTHING
                """,
                r,
            )
        counts["registered_model_tags"] = len(reg_model_tag_rows)

        model_ver_rows = cur_s.execute(
            """SELECT name, version, creation_time, last_updated_time, description,
                      user_id, current_stage, source, run_id, status, status_message,
                      run_link, storage_location, workspace
               FROM model_versions"""
        ).fetchall()
        for r in model_ver_rows:
            cur_p.execute(
                """
                INSERT INTO model_versions (name, version, creation_time, last_updated_time, description,
                                            user_id, current_stage, source, run_id, status, status_message,
                                            run_link, storage_location, workspace)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (workspace, name, version) DO NOTHING
                """,
                r,
            )
        counts["model_versions"] = len(model_ver_rows)

        model_ver_tag_rows = cur_s.execute(
            "SELECT key, value, name, version, workspace FROM model_version_tags"
        ).fetchall()
        for r in model_ver_tag_rows:
            cur_p.execute(
                """
                INSERT INTO model_version_tags (key, value, name, version, workspace)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (workspace, key, name, version) DO NOTHING
                """,
                r,
            )
        counts["model_version_tags"] = len(model_ver_tag_rows)

        reg_model_alias_rows = cur_s.execute(
            "SELECT alias, version, name, workspace FROM registered_model_aliases"
        ).fetchall()
        for r in reg_model_alias_rows:
            cur_p.execute(
                """
                INSERT INTO registered_model_aliases (alias, version, name, workspace)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (workspace, name, alias) DO NOTHING
                """,
                r,
            )
        counts["registered_model_aliases"] = len(reg_model_alias_rows)
    pg_conn.commit()

    # 8. Evaluation Datasets & Records
    print("8. Migrating evaluation_datasets & entity_associations...")
    with pg_conn.cursor() as cur_p:
        eval_ds_rows = cur_s.execute(
            """SELECT dataset_id, name, schema, profile, digest,
                      created_time, last_update_time, created_by, last_updated_by, workspace
               FROM evaluation_datasets"""
        ).fetchall()
        for r in eval_ds_rows:
            cur_p.execute(
                """
                INSERT INTO evaluation_datasets (dataset_id, name, schema, profile, digest,
                                                 created_time, last_update_time, created_by, last_updated_by, workspace)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (dataset_id) DO NOTHING
                """,
                r,
            )
        counts["evaluation_datasets"] = len(eval_ds_rows)

        eval_ds_tag_rows = cur_s.execute(
            "SELECT dataset_id, key, value FROM evaluation_dataset_tags"
        ).fetchall()
        for r in eval_ds_tag_rows:
            cur_p.execute(
                """
                INSERT INTO evaluation_dataset_tags (dataset_id, key, value)
                VALUES (%s, %s, %s)
                ON CONFLICT (dataset_id, key) DO NOTHING
                """,
                r,
            )
        counts["evaluation_dataset_tags"] = len(eval_ds_tag_rows)

        eval_ds_rec_rows = cur_s.execute(
            """SELECT dataset_record_id, dataset_id, inputs, expectations, tags,
                      source, source_id, source_type, created_time, last_update_time,
                      created_by, last_updated_by, input_hash, outputs
               FROM evaluation_dataset_records"""
        ).fetchall()
        for r in eval_ds_rec_rows:
            cur_p.execute(
                """
                INSERT INTO evaluation_dataset_records (dataset_record_id, dataset_id, inputs, expectations, tags,
                                                        source, source_id, source_type, created_time, last_update_time,
                                                        created_by, last_updated_by, input_hash, outputs)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (dataset_record_id) DO NOTHING
                """,
                r,
            )
        counts["evaluation_dataset_records"] = len(eval_ds_rec_rows)

        entity_assoc_rows = cur_s.execute(
            """SELECT association_id, source_type, source_id, destination_type, destination_id, created_time
               FROM entity_associations"""
        ).fetchall()
        for r in entity_assoc_rows:
            cur_p.execute(
                """
                INSERT INTO entity_associations (association_id, source_type, source_id, destination_type, destination_id, created_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_type, source_id, destination_type, destination_id) DO NOTHING
                """,
                r,
            )
        counts["entity_associations"] = len(entity_assoc_rows)
    pg_conn.commit()

    # 9. Metrics (Bulk streaming COPY with individual commit)
    print("9. Migrating metrics (bulk COPY)...")
    with pg_conn.cursor() as cur_p:
        cur_p.execute("SELECT count(*) FROM metrics")
        existing_metrics_count = cur_p.fetchone()[0]
        cur_s.execute("SELECT count(*) FROM metrics")
        total_metrics = cur_s.fetchone()[0]

        if existing_metrics_count < total_metrics:
            if existing_metrics_count > 0:
                print(
                    f"  Clearing {existing_metrics_count} existing metrics before clean bulk COPY..."
                )
                cur_p.execute("TRUNCATE TABLE metrics")
                pg_conn.commit()

            cur_s.execute(
                "SELECT key, value, timestamp, run_uuid, step, is_nan FROM metrics"
            )
            t0 = time.time()
            batch_size = 500_000
            copied = 0

            with tqdm(
                total=total_metrics, desc="Metrics bulk COPY", unit="rows"
            ) as pbar:
                with cur_p.copy(
                    "COPY metrics (key, value, timestamp, run_uuid, step, is_nan) FROM STDIN"
                ) as copy:
                    while True:
                        rows = cur_s.fetchmany(batch_size)
                        if not rows:
                            break
                        for r in rows:
                            copy.write_row((r[0], r[1], r[2], r[3], r[4], bool(r[5])))
                        copied += len(rows)
                        pbar.update(len(rows))

            t1 = time.time()
            print(
                f"  Copied {copied} metric rows in {t1 - t0:.2f}s ({(copied / max(0.001, t1 - t0)):.0f} rows/s)"
            )
            counts["metrics"] = copied
            pg_conn.commit()
        else:
            print(
                f"  Metrics already populated ({existing_metrics_count} rows). Skipping COPY."
            )
            counts["metrics"] = existing_metrics_count

    print("=== Step 1 Complete: All database metadata committed successfully! ===\n")
    return counts


def upload_run_artifacts(
    client: MlflowClient, run_uuid: str, exp_id: int, run_artifact_path: Path
) -> tuple[str, int, int]:
    """Upload all artifact files for a single run via MLflow REST API."""
    files = [f for f in run_artifact_path.rglob("*") if f.is_file()]
    if not files:
        return run_uuid, 0, 0

    total_bytes = sum(f.stat().st_size for f in files)
    try:
        # log_artifacts logs the entire contents of the directory into the root of run artifacts
        client.log_artifacts(run_uuid, str(run_artifact_path))
        return run_uuid, len(files), total_bytes
    except Exception as e:
        print(
            f"Error uploading artifacts for run {run_uuid} in exp {exp_id}: {e}",
            file=sys.stderr,
        )
        raise


def migrate_artifacts(client: MlflowClient, max_workers: int = 24) -> dict[str, Any]:
    """Migrate all artifacts to MLflow server via parallel multi-threading."""
    print("=== Step 2: Migrating Artifacts via MLflow API ===")
    tasks: list[tuple[str, int, Path]] = []

    for exp_id in [13, 14]:
        exp_dir = ARTIFACTS_ROOT / str(exp_id)
        if not exp_dir.exists():
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_uuid = run_dir.name
            artifacts_sub = run_dir / "artifacts"
            if artifacts_sub.is_dir() and any(artifacts_sub.iterdir()):
                tasks.append((run_uuid, exp_id, artifacts_sub))

    total_runs = len(tasks)
    print(
        f"Found {total_runs} runs with artifacts to upload across experiments 13 and 14."
    )

    uploaded_runs = 0
    uploaded_files = 0
    uploaded_bytes = 0
    t0 = time.time()

    # Disable mlflow's internal per-file tqdm to avoid cluttered terminal
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

    with tqdm(total=total_runs, desc="Uploading run artifacts", unit="run") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {
                executor.submit(
                    upload_run_artifacts, client, run_uuid, exp_id, path
                ): run_uuid
                for run_uuid, exp_id, path in tasks
            }
            for future in concurrent.futures.as_completed(future_to_run):
                run_uuid, num_files, byte_count = future.result()
                uploaded_runs += 1
                uploaded_files += num_files
                uploaded_bytes += byte_count
                pbar.update(1)

    t1 = time.time()
    elapsed = t1 - t0
    mb_uploaded = uploaded_bytes / (1024 * 1024)
    print(
        f"Uploaded {uploaded_runs} runs, {uploaded_files} files, {mb_uploaded:.2f} MB in {elapsed:.2f}s ({(mb_uploaded / max(0.001, elapsed)):.2f} MB/s)"
    )
    print("=== Step 2 Complete: All artifacts migrated successfully! ===\n")
    return {
        "uploaded_runs": uploaded_runs,
        "uploaded_files": uploaded_files,
        "uploaded_bytes": uploaded_bytes,
        "elapsed_seconds": elapsed,
    }


def verify_migration(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection[Any],
    client: MlflowClient,
) -> dict[str, Any]:
    """Verify database counts and MLflow API read capabilities."""
    print("=== Step 3: Verifying Migration Integrity ===")
    cur_s = sqlite_conn.cursor()
    cur_p = pg_conn.cursor()

    tables = [
        "experiments",
        "experiment_tags",
        "runs",
        "tags",
        "params",
        "latest_metrics",
        "metrics",
        "registered_models",
        "registered_model_tags",
        "model_versions",
        "model_version_tags",
        "registered_model_aliases",
        "evaluation_datasets",
        "evaluation_dataset_tags",
        "evaluation_dataset_records",
        "entity_associations",
    ]

    discrepancies: list[str] = []
    verification_stats: dict[str, Any] = {}

    for t in tables:
        cnt_s = cur_s.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        cur_p.execute(f'SELECT count(*) FROM "{t}"')
        cnt_p = cur_p.fetchone()[0]
        verification_stats[t] = {"sqlite": cnt_s, "postgres": cnt_p}
        status = "OK" if cnt_s == cnt_p else "MISMATCH"
        print(f"Table {t:30s}: SQLite={cnt_s:10d} | PG={cnt_p:10d} [{status}]")
        if cnt_s != cnt_p:
            discrepancies.append(f"Count mismatch on {t}: SQLite {cnt_s} vs PG {cnt_p}")

    # Verify MLflow API
    print("\nVerifying MLflow API responses:")
    experiments = client.search_experiments()
    print(f"  Found {len(experiments)} experiments in MLflow client:")
    for exp in experiments:
        print(
            f"    - ID: {exp.experiment_id}, Name: {exp.name}, Artifact Location: {exp.artifact_location}"
        )

    runs_ds1 = client.search_runs(["14"], max_results=5)
    runs_ds2 = client.search_runs(["13"], max_results=5)
    print(
        f"  Sample runs in ds1 (exp 14): {len(runs_ds1)} found, sample: {runs_ds1[0].info.run_id if runs_ds1 else 'none'}"
    )
    print(
        f"  Sample runs in ds2 (exp 13): {len(runs_ds2)} found, sample: {runs_ds2[0].info.run_id if runs_ds2 else 'none'}"
    )

    # Test artifact listing
    if runs_ds1:
        sample_run = runs_ds1[0].info.run_id
        artifacts = client.list_artifacts(sample_run)
        print(f"  Artifacts in sample run {sample_run}: {[a.path for a in artifacts]}")

    if discrepancies:
        raise RuntimeError(f"Verification failed with discrepancies: {discrepancies}")

    print("\n=== Step 3 Complete: All checks passed with 100% integrity! ===")
    return verification_stats


def main() -> None:
    print("Starting MLflow Migration to PostgreSQL & Port 5001...")
    start_time = time.time()

    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    pg_conn = get_pg_connection()
    client = MlflowClient(MLFLOW_SERVER_URL)

    db_counts = migrate_database(sqlite_conn, pg_conn)
    artifact_stats = migrate_artifacts(client, max_workers=24)
    verify_stats = verify_migration(sqlite_conn, pg_conn, client)

    total_time = time.time() - start_time
    report = {
        "status": "SUCCESS",
        "total_time_seconds": total_time,
        "database_counts": db_counts,
        "artifact_stats": artifact_stats,
        "verification": verify_stats,
    }

    report_path = Path("f1-migrate/migration_summary.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nMigration successfully completed in {total_time:.2f}s.")
    print(f"Summary written to {report_path}")


if __name__ == "__main__":
    main()
