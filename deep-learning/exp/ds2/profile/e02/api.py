"""DS2 e02 Word2Vec profile orchestration."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .modules import COMPONENTS, profile_modules
from .update import (
    CONDITIONS,
    ROOT,
    _load_data,
    _metadata,
    profile_condition,
)


DEFAULT_RESULTS = ROOT / "exp/ds2/profile/e02/results"


def run(
    *,
    devices: tuple[str, ...] = ("cpu", "cuda:0"),
    conditions: tuple[str, ...] | None = None,
    mode: str = "all",
    components: tuple[str, ...] | None = None,
    batch_size: int = 100,
    epochs: int = 10,
    update_warmup: int = 5,
    update_repetitions: int = 20,
    measured_updates: int = 1,
    module_warmup: int = 5,
    module_iterations: int = 20,
    output_dir: Path = DEFAULT_RESULTS,
) -> None:
    """Run the selected e02 profiles, or the complete CPU/GPU matrix."""
    if mode not in {"all", "update", "modules"}:
        raise ValueError(f"unknown profile mode: {mode}")
    selected_conditions = CONDITIONS if conditions is None else conditions
    unknown_conditions = set(selected_conditions) - set(CONDITIONS)
    if unknown_conditions:
        raise ValueError(
            f"unknown e02 profile conditions: {sorted(unknown_conditions)}"
        )
    unknown_components = set(components or ()) - set(COMPONENTS)
    if unknown_components:
        raise ValueError(
            f"unknown e02 profile components: {sorted(unknown_components)}"
        )

    for device in devices:
        backend, corpus, contexts, targets = _load_data(device)
        device_dir = output_dir / device.replace(":", "")
        device_dir.mkdir(parents=True, exist_ok=True)

        if mode in {"all", "update"}:
            update_rows = []
            for condition in selected_conditions:
                backend.seed(1)
                np.random.seed(1)
                result = profile_condition(
                    condition,
                    corpus=corpus,
                    contexts=contexts,
                    targets=targets,
                    backend=backend,
                    batch_size=batch_size,
                    epochs=epochs,
                    warmup_updates=update_warmup,
                    measured_updates=measured_updates,
                    phase_updates=0,
                    repetitions=update_repetitions,
                )
                update_rows.append(asdict(result))
                print(
                    f"[{device}] {condition} update: "
                    f"{result.mean_ms_per_update:.3f} ± "
                    f"{result.stdev_ms_per_update:.3f} ms; "
                    f"{result.estimated_seconds_per_epoch:.1f} ± "
                    f"{result.estimated_stdev_seconds_per_epoch:.1f} s/epoch; "
                    f"{result.estimated_seconds_total:.1f} ± "
                    f"{result.estimated_stdev_seconds_total:.1f} s total",
                    flush=True,
                )
            _write_payload(
                device_dir / "update.json",
                backend=backend,
                stage="update",
                results=update_rows,
                schema_version=3,
            )

        if mode in {"all", "modules"}:
            module_rows = []
            for condition in selected_conditions:
                backend.seed(1)
                np.random.seed(1)
                rows = profile_modules(
                    condition,
                    corpus=corpus,
                    contexts=contexts,
                    targets=targets,
                    backend=backend,
                    batch_size=batch_size,
                    components=components,
                    warmup_iterations=module_warmup,
                    measured_iterations=module_iterations,
                )
                module_rows.extend(rows)
                for row in rows:
                    timing = row["timing"]
                    assert isinstance(timing, dict)
                    print(
                        f"[{device}] {condition} {row['component']}: "
                        f"{float(timing['mean_ms']):.3f} ± "
                        f"{float(timing['stdev_ms']):.3f} ms",
                        flush=True,
                    )
            _write_payload(
                device_dir / "modules.json",
                backend=backend,
                stage="modules",
                results=module_rows,
                schema_version=1,
            )


def _write_payload(
    path: Path,
    *,
    backend,
    stage: str,
    results: list[dict[str, object]],
    schema_version: int,
) -> None:
    payload = {
        "schema_version": schema_version,
        "metadata": _metadata(backend, stage=stage),
        "results": results,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {path}", flush=True)
