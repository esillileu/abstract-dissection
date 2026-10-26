#!/usr/bin/env python3
"""One-time migration utility from legacy root folders to unified workspace layout.

Usage:
    # Dry-run inspection (default):
    uv run python tools/migrations/migrate_to_workspace.py

    # Apply migration (copies legacy paths into workspace):
    uv run python tools/migrations/migrate_to_workspace.py --apply
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def plan_migration(
    repo_root: Path, workspace_root: Path
) -> list[tuple[Path, Path, str]]:
    tasks: list[tuple[Path, Path, str]] = []

    # 1. Inputs (data/) -> workspace/inputs/
    data_dir = repo_root / "data"
    if data_dir.is_dir():
        for item in data_dir.iterdir():
            dest = workspace_root / "inputs" / item.name
            tasks.append((item, dest, "input_dataset"))

    # 2. Shared Prepared Data (phrase cache, vocab cache)
    cache_f2 = repo_root / ".cache/f2/w2v"
    if (cache_f2 / "phrases").is_dir():
        tasks.append(
            (
                cache_f2 / "phrases",
                workspace_root / "inputs/f2/prepared/phrases",
                "prepared_phrases",
            )
        )
    if (cache_f2 / "vocabulary").is_dir():
        tasks.append(
            (
                cache_f2 / "vocabulary",
                workspace_root / "inputs/f2/prepared/vocabulary",
                "prepared_vocabulary",
            )
        )

    # 3. MLflow Download Cache
    mlflow_cache = repo_root / ".cache/mlflow_artifact"
    if mlflow_cache.is_dir():
        tasks.append(
            (mlflow_cache, workspace_root / "analysis/cache/mlflow", "mlflow_cache")
        )

    # 4. Analysis Results (artifacts/analysis) -> workspace/analysis/results/
    artifacts_analysis = repo_root / "artifacts/analysis"
    if artifacts_analysis.is_dir():
        for item in artifacts_analysis.iterdir():
            dest = workspace_root / "analysis/results" / item.name
            tasks.append((item, dest, "analysis_results"))

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate repository data to unified workspace."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Execute file copies (default is dry-run)"
    )
    parser.add_argument(
        "--move", action="store_true", help="Move instead of copy (default is copy)"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root / "workspace"

    tasks = plan_migration(repo_root, workspace_root)
    print(f"Discovered {len(tasks)} migration mappings for workspace: {workspace_root}")

    for src, dst, kind in tasks:
        action_desc = "MOVE" if args.move else "COPY"
        if not args.apply:
            print(f"[DRY-RUN] {action_desc} ({kind}): {src} -> {dst}")
        else:
            print(f"Applying ({kind}): {src} -> {dst} ...")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if args.move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.is_file():
                if args.move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(src, dst)

    if not args.apply:
        print(
            "\nDry-run complete. No files were modified. Run with --apply to execute."
        )
    else:
        print("\nMigration complete.")


if __name__ == "__main__":
    main()
