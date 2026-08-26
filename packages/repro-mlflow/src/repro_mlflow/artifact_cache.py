"""Process-safe, disposable cache for MLflow artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from tqdm.auto import tqdm

_artifact_progress: ContextVar[object | None] = ContextVar(
    "mlprosection_artifact_progress", default=None
)


@contextmanager
def artifact_download_progress() -> Iterator[None]:
    """Show one aggregate progress bar for cache-backed downloads."""
    with tqdm(desc="Downloading analysis artifacts", unit="artifact") as progress:
        token = _artifact_progress.set(progress)
        try:
            yield
        finally:
            _artifact_progress.reset(token)


@contextmanager
def _mlflow_artifact_progress_disabled() -> Iterator[None]:
    """Hide MLflow's per-download progress bars.

    Callers of the cache may request several artifacts as part of one larger
    operation. MLflow creates a new progress bar for each download call, so
    showing those bars here produces noisy, misleading output. The operation
    owner remains responsible for presenting any aggregate progress.
    """
    variable = "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"
    previous = os.environ.get(variable)
    os.environ[variable] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def tracking_uri_key(uri: str) -> str:
    """Return a stable server identity without leaking URI credentials."""
    parsed = urlsplit(uri)
    host = parsed.hostname or ""
    netloc = host
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    normalized = urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path, parsed.query, "")
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class MlflowArtifactCache:
    def __init__(self, client, tracking_uri: str, root: Path | None = None) -> None:
        from repro_core.context.paths import WorkspacePaths

        self.client = client
        self.tracking_uri = tracking_uri
        self.server_key = tracking_uri_key(tracking_uri)
        paths = WorkspacePaths.from_environment(Path.cwd())
        self.root = (root or paths.cache_root / "mlflow_artifact").resolve()

    def path(self, run_id: str, artifact_path: str) -> Path:
        parts = tuple(
            part for part in Path(artifact_path).parts if part not in {"", "."}
        )
        _validate(run_id, parts)
        return self.root.joinpath(self.server_key, run_id, *parts)

    def get(self, run_id: str, artifact_path: str) -> Path:
        target = self.path(run_id, artifact_path)
        if target.exists():
            return target
        with self._lock(target):
            if target.exists():
                return target
            staged = self.fetch(run_id, artifact_path)
            try:
                self.publish(staged, target)
            finally:
                self.discard(staged)
        return target

    def fetch(self, run_id: str, artifact_path: str) -> Path:
        """Force a server download into an unpublished temporary directory."""
        target = self.path(run_id, artifact_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.download-", dir=target.parent)
        )
        try:
            with _mlflow_artifact_progress_disabled():
                downloaded = Path(
                    self.client.download_artifacts(
                        run_id, artifact_path, str(temporary)
                    )
                ).resolve()
            progress = _artifact_progress.get()
            if progress is not None:
                progress.update(1)
            if downloaded.is_relative_to(temporary.resolve()):
                return downloaded
            local = temporary / downloaded.name
            if downloaded.is_dir():
                shutil.copytree(downloaded, local)
            else:
                shutil.copy2(downloaded, local)
            return local
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def publish(self, source: Path, target: Path) -> None:
        """Atomically replace a cache entry with a fully downloaded artifact."""
        target.parent.mkdir(parents=True, exist_ok=True)
        incoming = target.parent / f".{target.name}.publish-{os.getpid()}"
        if incoming.exists():
            if incoming.is_dir():
                shutil.rmtree(incoming)
            else:
                incoming.unlink()
        if source.is_dir():
            shutil.copytree(source, incoming)
        else:
            shutil.copy2(source, incoming)
        backup = target.parent / f".{target.name}.previous-{os.getpid()}"
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(incoming, target)
        except Exception:
            if backup.exists():
                os.replace(backup, target)
            raise
        if backup.is_dir():
            shutil.rmtree(backup)
        elif backup.exists():
            backup.unlink()

    def replace(self, run_id: str, artifact_path: str, source: Path) -> Path:
        """Publish a force-downloaded, caller-verified value under the same lock."""
        target = self.path(run_id, artifact_path)
        with self._lock(target):
            self.publish(source, target)
        return target

    def discard(self, source: Path) -> None:
        """Remove the private download directory containing ``source``."""
        root = self.root.resolve()
        current = source.resolve()
        while current != root and current.is_relative_to(root):
            if ".download-" in current.name:
                shutil.rmtree(current, ignore_errors=True)
                return
            current = current.parent

    @contextmanager
    def _lock(self, target: Path) -> Iterator[None]:
        import fcntl

        lock_root = self.root / ".locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(str(target).encode("utf-8")).hexdigest()
        with (lock_root / f"{key}.lock").open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate(run_id: str, parts: tuple[str, ...]) -> None:
    values = (run_id, *parts)
    if not parts or any(
        not value or value in {".", ".."} or Path(value).name != value
        for value in values
    ):
        raise ValueError("invalid MLflow artifact coordinate")


__all__ = ["MlflowArtifactCache", "tracking_uri_key"]
