"""Execute one vendored original trial and project its raw record."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from deepscratch.core import BackendConfig, make_backend

from dlfs.original_runtime.runtime_context import reset_runtime, set_runtime
from repro_core.context.contracts import ExperimentResult


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


def _promote_final_checkpoint(
    config: dict[str, object], context, output: Path, final: dict[str, float]
) -> Path | None:
    """Publish an original parameter archive through the canonical checkpoint API."""
    checkpoint = config.get("checkpoint", {})
    if not isinstance(checkpoint, dict) or not checkpoint.get("save_final", False):
        return None
    source = output / "checkpoint.npz"
    if not source.is_file():
        raise ValueError("checkpoint.save_final requires raw/checkpoint.npz")
    root = Path(str(context.metadata["checkpoint_root"]))
    root.mkdir(parents=True, exist_ok=True)
    target = root / "final.npz"
    with tempfile.NamedTemporaryFile(dir=root, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        import numpy as np

        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        # Original Word2Vec archives call the analysis-ready embedding matrix
        # ``word_vectors`` and retain the exact upstream parameter sequence as
        # ``param_###``.  Keep both and expose the canonical model key.
        if "W_in" not in arrays and "word_vectors" in arrays:
            arrays["W_in"] = arrays["word_vectors"]
        np.savez(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    pointer = {
        "schema_version": 2,
        "role": "latest",
        "path": target.name,
        "sha256": digest,
        "epoch": int(final.get("final/system/completed_epochs", 0)),
        "update": int(final.get("final/system/total_updates", 0)),
    }
    temporary_pointer = root / ".latest.json.tmp"
    temporary_pointer.write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_pointer, root / "latest.json")
    return target


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


def _metric_rows(
    root: Path, *, default_accuracy_split: str | None = None
) -> tuple[list[tuple[int, str, float]], dict[str, float]]:
    path = root / "metrics.csv"
    if not path.is_file():
        return [], {}
    output: list[tuple[int, str, float]] = []
    final: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if row.get("metric") and row.get("value") not in {None, ""}:
                pairs = [(str(row["metric"]), row["value"])]
            else:
                pairs = [
                    (key, value)
                    for key, value in row.items()
                    if key
                    not in {
                        "update",
                        "epoch",
                        "plot_index",
                        "condition",
                        "batch_size",
                        "eval_interval",
                    }
                ]
            for key, value in pairs:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                split = str(row.get("split", ""))
                if not split and "accuracy" in key.lower():
                    split = default_accuracy_split or ""
                metric = _metric_name(key, split=split)
                if metric.endswith("/accuracy"):
                    # DS1 original e03/e04 record accuracy once per epoch,
                    # while their update counter advances three times per
                    # epoch.  Preserve the graph's epoch axis in the metric
                    # series; zero is a valid first epoch and must not fall
                    # through via truthiness-based selection.
                    step = _int_value(
                        row.get("epoch"),
                        row.get("update"),
                        row.get("plot_index"),
                        default=index,
                    )
                else:
                    step = _int_value(
                        row.get("update"),
                        row.get("epoch"),
                        row.get("plot_index"),
                        default=index,
                    )
                output.append((step, metric, number))
                final[f"final/{metric}"] = number
    return output, final


def _metric_name(key: str, *, split: str = "") -> str:
    lowered = key.lower()
    if "perplexity" in lowered:
        prefix = (
            split.lower() if split.lower() in {"train", "valid", "test"} else "train"
        )
        return f"{prefix}/perplexity"
    if "accuracy" in lowered:
        prefix = (
            split.lower() if split.lower() in {"train", "test", "test-full"} else "test"
        )
        return f"{prefix}/accuracy"
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
