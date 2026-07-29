"""Runtime and model-size measurements for original-code runners."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter
from typing import Iterator, Mapping, Sequence


MEASUREMENT_SCHEMA_VERSION = 1


def _synchronize() -> None:
    try:
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
    except (ImportError, RuntimeError):
        pass


class OriginalMeasurements:
    """Accumulate synchronized training wall time and persist model size."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.training_wall_time_s = 0.0

    @contextmanager
    def training(self) -> Iterator[None]:
        _synchronize()
        started = perf_counter()
        try:
            yield
        finally:
            _synchronize()
            self.training_wall_time_s += perf_counter() - started

    def save(
        self,
        params: Mapping[str, object] | Sequence[object],
        *,
        scope: str = "optimizer_updates",
    ) -> None:
        parameter_count, tensor_count, shared_references = count_parameters(params)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "timing.json").write_text(
            json.dumps(
                {
                    "schema_version": MEASUREMENT_SCHEMA_VERSION,
                    "scope": scope,
                    "training_wall_time_s": self.training_wall_time_s,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.output / "parameter_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": MEASUREMENT_SCHEMA_VERSION,
                    "parameter_count": parameter_count,
                    "tensor_count": tensor_count,
                    "shared_reference_count": shared_references,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def count_parameters(
    params: Mapping[str, object] | Sequence[object],
) -> tuple[int, int, int]:
    values = list(params.values()) if isinstance(params, Mapping) else list(params)
    unique = []
    seen = set()
    for value in values:
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(value)
    parameter_count = sum(int(getattr(value, "size")) for value in unique)
    return parameter_count, len(unique), len(values) - len(unique)


__all__ = ["OriginalMeasurements", "count_parameters"]
