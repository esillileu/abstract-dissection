"""The only public gateway to retired DeepScratch storage formats."""

from __future__ import annotations

from collections.abc import Iterable

from exp.framework.results import NativeRunResult
from mlprosection_mlflow.artifact_cache import MlflowArtifactCache

from ..identity import Variant, Volume
from ..analysis.declarations import MetricDeclaration
from .attempts import load_legacy_attempts
from .result_adapter import load_legacy_result


class LegacyCompatibility:
    """Contain every read of retired namespaces and result layouts."""

    def __init__(self, client, *, artifact_cache: MlflowArtifactCache | None = None) -> None:
        self._client = client
        self._artifact_cache = artifact_cache

    def attempts(
        self,
        volume: Volume,
        variant: Variant,
    ) -> list[dict[str, object]]:
        return load_legacy_attempts(self._client, volume, variant)

    def load_result(
        self,
        run_id: str,
        *,
        variant: Variant,
        declarations: Iterable[MetricDeclaration],
    ) -> NativeRunResult:
        return load_legacy_result(
            self._client,
            run_id,
            variant=variant,
            declarations=declarations,
            artifact_cache=self._artifact_cache,
        )
