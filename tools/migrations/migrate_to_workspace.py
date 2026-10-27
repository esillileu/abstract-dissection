#!/usr/bin/env python3
"""One-time migration utility from legacy root folders to unified workspace layout.

Usage:
    # Dry-run inspection (default):
    uv run python tools/migrations/migrate_to_workspace.py

    # Apply migration with move (fastest, atomic within filesystem):
    uv run python tools/migrations/migrate_to_workspace.py --apply --move

    # Apply migration with copy (preserves legacy directories):
    uv run python tools/migrations/migrate_to_workspace.py --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def plan_migration(
    repo_root: Path, workspace_root: Path
) -> list[tuple[Path, Path, str]]:
    tasks: list[tuple[Path, Path, str]] = []

    # 1. Inputs (data/) -> workspace/inputs/
    data_dir = repo_root / "data"
    if data_dir.is_dir():
        for item in sorted(data_dir.iterdir()):
            if item.name.startswith("."):
                continue
            dest = workspace_root / "inputs" / item.name
            tasks.append((item, dest, "input_dataset"))

    # 2. Shared Prepared & Reference Data
    cache_f2_w2v = repo_root / ".cache/f2/w2v"
    if (cache_f2_w2v / "phrases").is_dir():
        tasks.append(
            (
                cache_f2_w2v / "phrases",
                workspace_root / "inputs/f2/prepared/phrases",
                "prepared_phrases",
            )
        )
    if (cache_f2_w2v / "vocabulary").is_dir():
        tasks.append(
            (
                cache_f2_w2v / "vocabulary",
                workspace_root / "inputs/f2/prepared/vocabulary",
                "prepared_vocabulary",
            )
        )
    if (repo_root / ".cache/f2/pochih_repo").is_dir():
        tasks.append(
            (
                repo_root / ".cache/f2/pochih_repo",
                workspace_root / "inputs/f2/references/pochih_repo",
                "reference_repo",
            )
        )

    # 3. MLflow Download Cache & Raw Cache
    mlflow_cache = repo_root / ".cache/mlflow_artifact"
    if mlflow_cache.is_dir():
        tasks.append(
            (mlflow_cache, workspace_root / "analysis/cache/mlflow", "mlflow_cache")
        )
    mlflow_raw = repo_root / ".cache/mlflow_raw"
    if mlflow_raw.is_dir():
        tasks.append(
            (mlflow_raw, workspace_root / "analysis/cache/mlflow_raw", "mlflow_raw")
        )

    # 4. Analysis Results (artifacts/analysis) -> workspace/analysis/results/
    artifacts_analysis = repo_root / "artifacts/analysis"
    if artifacts_analysis.is_dir():
        for item in sorted(artifacts_analysis.iterdir()):
            if item.name.startswith("."):
                continue
            dest = workspace_root / "analysis/results" / item.name
            tasks.append((item, dest, "analysis_results"))

    # 5. Derived Analysis Cache (.cache/exp/ & .cache/f2/w2v1/)
    cache_exp_ds = repo_root / ".cache/exp/deepscratch"
    if cache_exp_ds.is_dir():
        tasks.append(
            (
                cache_exp_ds,
                workspace_root / "analysis/cache/deepscratch",
                "cache_exp_ds",
            )
        )
    cache_exp_f2 = repo_root / ".cache/exp/f2"
    if cache_exp_f2.is_dir():
        tasks.append(
            (cache_exp_f2, workspace_root / "analysis/cache/f2", "cache_exp_f2")
        )
    cache_f2_w2v1 = repo_root / ".cache/f2/w2v1"
    if cache_f2_w2v1.is_dir():
        tasks.append(
            (
                cache_f2_w2v1,
                workspace_root / "analysis/cache/results/f2/w2v1",
                "analysis_eval_cache",
            )
        )

    # 6. Staging / Execution Runs (.staging/exp/) -> workspace/runs/
    staging_exp = repo_root / ".staging/exp"
    if staging_exp.is_dir():
        for item in sorted(staging_exp.iterdir()):
            if item.name.startswith("."):
                continue
            dest = workspace_root / "runs" / item.name
            tasks.append((item, dest, "staging_runs"))

    return tasks


def merge_transfer(src: Path, dst: Path, *, move: bool) -> None:
    """Recursively transfer files from src into dst, merging directories if dst already exists."""
    if not src.exists():
        return

    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(dst))
        else:
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        return

    if src.is_file():
        if dst.is_dir():
            raise RuntimeError(f"Cannot overwrite directory {dst} with file {src}")
        if move:
            os.replace(src, dst)
        else:
            shutil.copy2(src, dst)
        return

    if src.is_dir():
        if dst.is_file():
            raise RuntimeError(f"Cannot overwrite file {dst} with directory {src}")
        for item in list(src.iterdir()):
            merge_transfer(item, dst / item.name, move=move)
        if move:
            try:
                src.rmdir()
            except OSError:
                pass


def setup_backward_compatibility(repo_root: Path, workspace_root: Path) -> None:
    """Set up symlinks for tools or legacy scripts expecting former paths."""
    # 1. workspace/analysis/cache/mlflow_artifact -> mlflow
    mlflow_dir = workspace_root / "analysis/cache/mlflow"
    mlflow_artifact = workspace_root / "analysis/cache/mlflow_artifact"
    if mlflow_dir.is_dir():
        if mlflow_artifact.is_symlink():
            mlflow_artifact.unlink()
        elif mlflow_artifact.is_dir():
            # Merge any remnants into mlflow
            merge_transfer(mlflow_artifact, mlflow_dir, move=True)
            try:
                mlflow_artifact.rmdir()
            except OSError:
                pass
        if not mlflow_artifact.exists():
            mlflow_artifact.symlink_to("mlflow", target_is_directory=True)
            print("Created symlink: workspace/analysis/cache/mlflow_artifact -> mlflow")

    # 2. data/ -> symlinks to workspace/inputs/ items
    data_dir = repo_root / "data"
    inputs_dir = workspace_root / "inputs"
    if inputs_dir.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)
        for item in inputs_dir.iterdir():
            target_link = data_dir / item.name
            if not target_link.exists():
                try:
                    rel_target = Path("..") / "workspace" / "inputs" / item.name
                    target_link.symlink_to(
                        rel_target, target_is_directory=item.is_dir()
                    )
                except OSError:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate repository data to unified workspace."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Execute migration (default is dry-run)"
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
            merge_transfer(src, dst, move=args.move)

    if not args.apply:
        print(
            "\nDry-run complete. No files were modified. Run with --apply to execute."
        )
    else:
        setup_backward_compatibility(repo_root, workspace_root)
        print("\nMigration complete.")


if __name__ == "__main__":
    main()
