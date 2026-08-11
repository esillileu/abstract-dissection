"""Execute one vendored original trial and project its raw record."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.experiment.contracts import ExperimentResult

from exp.original.runtime_context import reset_runtime, set_runtime


class _ModelRecord:
    def __init__(self, device: str, artifact_root: Path) -> None:
        self.backend = make_backend(BackendConfig(device=device, dtype="float32", seed=0))
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


def execute(config: dict[str, object], context, *, domain: str, source_root: Path) -> ExperimentResult:
    experiment = str(config["source_experiment"])
    trial_id = str(config["trial_id"])
    module = importlib.import_module(f"exp.{domain.removesuffix('_original')}.original.run.{experiment}")
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
        if domain == "ds1_original":
            trial.runner(source_root, output)
        else:
            trial.runner(source_root, output, _dependency_root(config, output))
    finally:
        reset_runtime(tokens)
    wall = perf_counter() - started
    rows, final = _metric_rows(output)
    final.update({
        "runtime/train_total_s": _training_time(output, wall),
        "final/system/total_updates": float(max((row[0] for row in rows), default=0)),
        "final/system/completed_epochs": float(_last_axis(output, "epoch")),
        "final/system/samples_seen": 0.0,
    })
    provenance = {
        "domain": domain,
        "master_seed": seed,
        "device": selected_device,
        "upstream": json.loads((source_root.parent / "PROVENANCE.json").read_text(encoding="utf-8")),
    }
    (output / "run_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    context.metadata["upstream_provenance"] = provenance
    return ExperimentResult(final, output, model=_ModelRecord(selected_device, output), artifacts=tuple(output.iterdir()), metric_rows=tuple(rows))


def _dependency_root(config: dict[str, object], output: Path) -> Path:
    checkpoint = config.get("checkpoint", {})
    if not isinstance(checkpoint, dict) or not checkpoint.get("source_path"):
        return output.parent
    source = Path(str(checkpoint["source_path"]))
    root = output / "dependency"
    target = root / "data" / "e07" / "dlfs2.ch08.date.attention-seq2seq-reverse"
    target.mkdir(parents=True, exist_ok=True)
    candidate = source / "checkpoint.npz" if source.is_dir() else source
    if not candidate.is_file():
        raise ValueError(f"e08 source artifact is missing checkpoint.npz: {source}")
    # The old e08 adapter only requires the archive and a valid-cache marker is
    # intentionally bypassed by its promoted-domain branch.
    (target / "SOURCE_PATH").write_text(str(candidate), encoding="utf-8")
    return root


def _metric_rows(root: Path) -> tuple[list[tuple[int, str, float]], dict[str, float]]:
    path = root / "metrics.csv"
    if not path.is_file():
        return [], {}
    output: list[tuple[int, str, float]] = []
    final: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            step = _int_value(row.get("update"), row.get("epoch"), row.get("plot_index"), default=index)
            if row.get("metric") and row.get("value") not in {None, ""}:
                pairs = [(str(row["metric"]), row["value"])]
            else:
                pairs = [(key, value) for key, value in row.items() if key not in {"update", "epoch", "plot_index", "condition", "batch_size", "eval_interval"}]
            for key, value in pairs:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                metric = _metric_name(key)
                output.append((step, metric, number))
                final[f"final/{metric}"] = number
    return output, final


def _metric_name(key: str) -> str:
    lowered = key.lower()
    if "perplexity" in lowered:
        return "train/perplexity"
    if "accuracy" in lowered:
        return "test/accuracy"
    if "loss" in lowered or "objective" in lowered:
        return "train/loss"
    return f"observation/{key}"


def _int_value(*values, default: int) -> int:
    for value in values:
        if value not in {None, ""}:
            return int(float(value))
    return default


def _last_axis(root: Path, name: str) -> int:
    path = root / "metrics.csv"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as stream:
        values = [row.get(name) for row in csv.DictReader(stream)]
    return max((_int_value(value, default=0) for value in values), default=0)


def _training_time(root: Path, fallback: float) -> float:
    path = root / "timing.json"
    if not path.is_file():
        return fallback
    return float(json.loads(path.read_text(encoding="utf-8"))["training_wall_time_s"])


def _checkpoint_parameters(root: Path):
    path = root / "checkpoint.npz"
    if not path.is_file():
        return ()
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return tuple(
            (name.removeprefix("param__"), archive[name].copy())
            for name in archive.files
            if name.startswith("param_")
        )
