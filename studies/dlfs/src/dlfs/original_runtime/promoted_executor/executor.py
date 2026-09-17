from __future__ import annotations

import importlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from deepscratch.core import BackendConfig, make_backend

from dlfs.original_runtime.runtime_context import reset_runtime, set_runtime
from repro_core.context.contracts import ExperimentResult

from .checkpoint import (
    _checkpoint_parameters,
    _dependency_root,
    _promote_final_checkpoint,
)
from .metrics import (
    _last_axis,
    _metric_rows,
    _training_time,
)


class _ModelRecord:
    def __init__(self, device: str, artifact_root: Path) -> None:
        self.backend = make_backend(
            BackendConfig(device=device, dtype="float32", seed=0)
        )
        self._parameters = _checkpoint_parameters(artifact_root)

    def named_parameters(self):
        return tuple(
            (
                name,
                SimpleNamespace(
                    data=self.backend.asarray(value),
                    backend=self.backend,
                    requires_grad=True,
                ),
            )
            for name, value in self._parameters
        )

    def __str__(self) -> str:
        return "Vendored upstream model; see parameter_manifest.json raw artifact"


def execute(
    config: dict[str, object], context, *, domain: str, source_root: Path
) -> ExperimentResult:
    experiment = str(config["source_experiment"])
    trial_id = str(config["trial_id"])
    try:
        _, volume, variant = domain.split(".")
    except ValueError as exc:
        raise ValueError(f"invalid canonical execution identity: {domain}") from exc
    if variant != "original":
        raise ValueError(f"original executor received non-original identity: {domain}")
    module = importlib.import_module(f"dlfs.{volume}.original.run.{experiment}")
    trial = next((item for item in module.TRIALS if item.trial_id == trial_id), None)
    if trial is None:
        raise ValueError(f"unknown original trial: {experiment}/{trial_id}")
    numerics = config.get("numerics", {})
    assert isinstance(numerics, dict)
    selected_device = str(numerics.get("device", "cpu"))
    seed = int(config.get("seed", 1))
    output = Path(str(context.metadata["artifact_root"])) / "raw"
    output.mkdir(parents=True, exist_ok=True)
    tokens = set_runtime(seed=seed, selected_device=selected_device, config=config)
    started = perf_counter()
    try:
        if domain == "deepscratch.ds1.original":
            trial.runner(source_root, output)
        else:
            trial.runner(source_root, output, _dependency_root(config, output))
    finally:
        reset_runtime(tokens)
    wall = perf_counter() - started
    default_accuracy_split = (
        "train"
        if domain == "deepscratch.ds1.original" and experiment == "e05"
        else None
    )
    rows, final = _metric_rows(output, default_accuracy_split=default_accuracy_split)
    final.update(
        {
            "runtime/train_total_s": _training_time(output, wall),
            "final/system/total_updates": float(
                max((row[0] for row in rows), default=0)
            ),
            "final/system/completed_epochs": float(_last_axis(output, "epoch")),
            "final/system/samples_seen": 0.0,
        }
    )
    _promote_final_checkpoint(config, context, output, final)
    provenance = {
        "domain": domain,
        "master_seed": seed,
        "device": selected_device,
        "upstream": json.loads(
            (source_root.parent / "provenance.json").read_text(encoding="utf-8")
        ),
    }
    (output / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    context.metadata["upstream_provenance"] = provenance
    return ExperimentResult(
        final,
        output,
        model=_ModelRecord(selected_device, output),
        artifacts=tuple(output.iterdir()),
        metric_rows=tuple(rows),
    )
