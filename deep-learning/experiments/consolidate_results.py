"""Consolidate legacy local results into the owning experiment directory.

This deliberately leaves ``experiments/data`` untouched: it contains the
MLflow tracking database and the server-managed artifact store.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


EXPERIMENTS_ROOT = Path("experiments")
LEGACY_RESULTS_ROOT = EXPERIMENTS_ROOT / "results"


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _move_file(source: Path, target: Path, *, apply: bool) -> None:
    if target.exists():
        if source.is_file() and target.is_file() and _digest(source) == _digest(target):
            print(f"remove duplicate {source}")
            if apply:
                source.unlink()
            return
        raise ValueError(f"refusing to overwrite conflicting result: {target}")
    print(f"{source} -> {target}")
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))


def _move_children(source: Path, target: Path, *, apply: bool) -> None:
    if not source.is_dir():
        return
    for child in sorted(source.iterdir()):
        _move_file(child, target / child.name, apply=apply)
    if apply and source.exists() and not any(source.iterdir()):
        source.rmdir()


def _move_checkpoints(source: Path, target: Path, *, apply: bool) -> None:
    if not source.is_dir():
        return
    for run_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        checkpoint = run_dir / "final.npz"
        if not checkpoint.is_file():
            if not any(run_dir.iterdir()):
                print(f"remove stale empty checkpoint directory {run_dir}")
                if apply:
                    run_dir.rmdir()
                continue
            raise ValueError(f"expected final checkpoint in {run_dir}")
        _move_file(checkpoint, target / run_dir.name / checkpoint.name, apply=apply)
        if apply and run_dir.exists() and not any(run_dir.iterdir()):
            run_dir.rmdir()
    if apply and source.exists() and not any(source.iterdir()):
        source.rmdir()


def _remove_stale(path: Path, *, apply: bool) -> None:
    if not path.exists():
        return
    print(f"remove stale {path}")
    if apply:
        shutil.rmtree(path)


def _remove_empty(path: Path, *, apply: bool) -> None:
    if not path.is_dir() or any(path.iterdir()):
        return
    print(f"remove empty {path}")
    if apply:
        path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="deepbase1")
    parser.add_argument("--apply", action="store_true", help="perform moves and removals")
    args = parser.parse_args()

    domain_root = EXPERIMENTS_ROOT / args.domain / "results"
    legacy_domain_root = LEGACY_RESULTS_ROOT / args.domain
    _move_children(LEGACY_RESULTS_ROOT / "mlflow_artifacts", domain_root / "mlflow_artifacts", apply=args.apply)
    _move_children(LEGACY_RESULTS_ROOT / "analysis", domain_root / "analysis", apply=args.apply)
    _move_children(LEGACY_RESULTS_ROOT / "run_logs", domain_root / "run_logs", apply=args.apply)
    _move_checkpoints(LEGACY_RESULTS_ROOT / "checkpoints" / args.domain, domain_root / "checkpoints", apply=args.apply)
    _move_children(legacy_domain_root / "mlflow_artifacts", domain_root / "mlflow_artifacts", apply=args.apply)
    _move_children(legacy_domain_root / "analysis", domain_root / "analysis", apply=args.apply)
    _move_children(legacy_domain_root / "run_logs", domain_root / "run_logs", apply=args.apply)
    _move_checkpoints(legacy_domain_root / "checkpoints", domain_root / "checkpoints", apply=args.apply)
    _remove_empty(LEGACY_RESULTS_ROOT / "checkpoints", apply=args.apply)
    _remove_empty(legacy_domain_root, apply=args.apply)
    _remove_empty(LEGACY_RESULTS_ROOT, apply=args.apply)
    _remove_stale(LEGACY_RESULTS_ROOT / "mlflow", apply=args.apply)
    _remove_stale(LEGACY_RESULTS_ROOT / "runs", apply=args.apply)


if __name__ == "__main__":
    main()
