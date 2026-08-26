"""Prepare disposable worktrees at the immutable upstream commits."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def head_commit(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD")


def source_hashes(repo: Path, paths: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in paths:
        content = subprocess.run(
            ("git", "-C", str(repo), "show", f"HEAD:{relative}"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        import hashlib

        hashes[relative] = hashlib.sha256(content).hexdigest()
    return hashes


@contextmanager
def clean_worktree(repo: Path) -> Iterator[Path]:
    """Yield a detached worktree and leave the user's nested checkout untouched."""
    root = Path(tempfile.mkdtemp(prefix="original-source-"))
    checkout = root / "checkout"
    commit = head_commit(repo)
    subprocess.run(
        (
            "git",
            "-C",
            str(repo),
            "worktree",
            "add",
            "--quiet",
            "--detach",
            str(checkout),
            commit,
        ),
        check=True,
    )
    try:
        _seed_untracked_dataset_cache(repo, checkout)
        yield checkout
    finally:
        subprocess.run(
            ("git", "-C", str(repo), "worktree", "remove", "--force", str(checkout)),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(root, ignore_errors=True)


def _seed_untracked_dataset_cache(repo: Path, checkout: Path) -> None:
    """Copy only known download caches; never copy modified source modules."""
    for relative in (
        "dataset/mnist.pkl",
        "dataset/ptb.train.npy",
        "dataset/ptb.valid.npy",
        "dataset/ptb.test.npy",
    ):
        source = repo / relative
        target = checkout / relative
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
