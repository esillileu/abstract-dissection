"""Analysis cache and result downloader for F2.

Implements ref.md section 4.1 & 4.7:
- Isolated MLflow download cache: workspace/analysis/cache/mlflow/<server_key>/<run_id>/
- Local intermediate evaluation/result cache: workspace/analysis/cache/results/
- Separates download/caching from suite-specific table mapping
"""

from __future__ import annotations

from pathlib import Path

from mlflow import MlflowClient

from repro_core.context.paths import RuntimePaths
from repro_mlflow.artifact_cache import MlflowArtifactCache


class AnalysisCacheManager:
    """Manages MLflow downloads and statistical evaluation cache for F2."""

    def __init__(self, tracking_uri: str, paths: RuntimePaths | None = None) -> None:
        self.tracking_uri = tracking_uri
        self._paths = paths or RuntimePaths.from_environment()
        self.client = MlflowClient(tracking_uri=tracking_uri)

        cache_root = getattr(
            self._paths, "mlflow_cache_root", self._paths.cache_root / "mlflow"
        )
        self._mlflow_cache = MlflowArtifactCache(
            self.client, tracking_uri, root=cache_root
        )

    def download_run_artifacts(
        self,
        run_id: str,
        artifact_path: str = "",
    ) -> Path:
        """Download and cache artifacts for a specific MLflow run."""
        return self._mlflow_cache.download(run_id, artifact_path=artifact_path)

    def get_analysis_cache_dir(self, suite_id: str) -> Path:
        """Get intermediate results cache directory."""
        if hasattr(self._paths, "analysis_cache"):
            cache_dir = self._paths.analysis_cache("f2", suite_id)
        else:
            cache_dir = self._paths.cache_root / "f2" / suite_id / "results"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def get_analysis_results_dir(self, suite_id: str, table: str | None = None) -> Path:
        """Get final analysis deliverable directory."""
        base = self._paths.workspace_root / "analysis" / "results" / "f2" / suite_id
        if table:
            base = base / table
        base.mkdir(parents=True, exist_ok=True)
        return base


__all__ = ["AnalysisCacheManager"]
