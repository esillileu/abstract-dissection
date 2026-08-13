"""Experiment-owned cross-variant comparison table rendering."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from mlflow.tracking import MlflowClient

from .identity import Variant, Volume, legacy_namespace


@dataclass(frozen=True)
class MetricDeclaration:
    metric_id: str
    unit: str
    implemented: tuple[str, ...]
    original: tuple[str, ...]


TEST_ACCURACY = MetricDeclaration(
    "test_accuracy", "fraction",
    ("final/test/accuracy", "test/accuracy"),
    ("final/test/accuracy", "test/accuracy"),
)
TRAIN_PERPLEXITY = MetricDeclaration(
    "train_perplexity", "perplexity",
    ("final/train/perplexity", "train/perplexity"),
    ("final/train/perplexity", "train/perplexity"),
)
TEST_PERPLEXITY = MetricDeclaration(
    "test_perplexity", "perplexity",
    ("final/test/perplexity", "test/perplexity"),
    ("final/test/perplexity", "test/perplexity"),
)
TEST_EXACT_MATCH = MetricDeclaration(
    "test_exact_match", "percent",
    ("final/test/exact_match", "test/exact_match"),
    ("final/test/accuracy", "test/accuracy"),
)


def write_comparison_table(
    tracking_uri: str,
    *,
    volume: Volume,
    experiment_ids: list[str],
    output_dir: Path,
) -> Path:
    client = MlflowClient(tracking_uri=tracking_uri)
    selected = set(experiment_ids)
    runs = {
        variant: _canonical_runs(client, volume, variant, selected)
        for variant in Variant
    }
    coordinates = sorted(
        set(runs[Variant.IMPLEMENTED]) | set(runs[Variant.ORIGINAL])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "comparison.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "experiment_id", "condition_id", "seed", "metric_id", "unit",
                "protocol_version", "implemented_run_id", "implemented_value",
                "implemented_availability", "original_run_id", "original_value",
                "original_availability",
            ),
        )
        writer.writeheader()
        for coordinate in coordinates:
            for declaration in _declarations(volume, coordinate[0]):
                left = runs[Variant.IMPLEMENTED].get(coordinate)
                right = runs[Variant.ORIGINAL].get(coordinate)
                left_value = _metric(left, declaration.implemented)
                right_value = _metric(right, declaration.original)
                writer.writerow({
                    "experiment_id": coordinate[0],
                    "condition_id": coordinate[1],
                    "seed": coordinate[2],
                    "metric_id": declaration.metric_id,
                    "unit": declaration.unit,
                    "protocol_version": _protocol(left, right),
                    "implemented_run_id": "" if left is None else left.info.run_id,
                    "implemented_value": "" if left_value is None else left_value,
                    "implemented_availability": "available" if left_value is not None else "unavailable",
                    "original_run_id": "" if right is None else right.info.run_id,
                    "original_value": "" if right_value is None else right_value,
                    "original_availability": "available" if right_value is not None else "unavailable",
                })
    return output


def _canonical_runs(client, volume, variant, selected):
    output = {}
    for namespace in (f"deepscratch.{volume.value}", legacy_namespace(volume, variant)):
        experiment = client.get_experiment_by_name(namespace)
        if experiment is None:
            continue
        filter_string = "attributes.status = 'FINISHED' and tags.`run.type` = 'seed_trial'"
        if namespace.startswith("deepscratch."):
            filter_string += f" and tags.`implementation.variant` = '{variant.value}'"
        candidates = client.search_runs(
            [experiment.experiment_id], filter_string=filter_string,
            order_by=["attributes.start_time DESC"], max_results=10_000,
        )
        for run in candidates:
            if run.data.tags.get("transfer.import.disposition") == "imported-alternate":
                continue
            experiment_id = run.data.tags.get("experiment.id") or run.data.tags.get("experiment.ids", "").split(",")[0]
            if selected and experiment_id not in selected:
                continue
            native_condition = run.data.tags.get("condition.id") or run.data.tags.get("atomic_run.id", "")
            condition = run.data.tags.get("comparison.condition_id") or _condition_id(
                volume, experiment_id, native_condition
            )
            seed = run.data.tags.get("master_seed") or run.data.params.get("seed/master", "single")
            output.setdefault((experiment_id, condition, str(seed)), run)
    return output


def _metric(run, candidates):
    if run is None:
        return None
    for name in candidates:
        if name in run.data.metrics:
            return run.data.metrics[name]
    return None


def _protocol(left, right):
    values = {
        run.data.tags.get("protocol.version", "legacy")
        for run in (left, right) if run is not None
    }
    if values <= {"legacy", "book-source-v1"}:
        return "book-source-v1"
    return values.pop() if len(values) == 1 else "protocol-mismatch"


def _condition_id(volume, experiment_id, native):
    if volume is not Volume.DS2:
        return native
    aliases = {
        ("e03", "LM-SMALL-RNN"): "small-rnnlm",
        ("e03", "SMALL-RNNLM"): "small-rnnlm",
        ("e04", "LM-LSTM"): "lstm-rnnlm",
        ("e04", "LSTM-RNNLM"): "lstm-rnnlm",
        ("e05", "LM-BETTER-RECIPE"): "better-rnnlm",
        ("e05", "BETTER-RNNLM"): "better-rnnlm",
        ("e06", "SEQA-VAN-FWD"): "seq2seq-forward",
        ("e06", "SEQ2SEQ-FORWARD"): "seq2seq-forward",
        ("e06", "SEQA-VAN-REV"): "seq2seq-reverse",
        ("e06", "SEQ2SEQ-REVERSE"): "seq2seq-reverse",
        ("e06", "SEQA-PEEKY-FWD"): "peeky-forward",
        ("e06", "PEEKY-FORWARD"): "peeky-forward",
        ("e06", "SEQA-PEEKY-REV"): "peeky-reverse",
        ("e06", "PEEKY-REVERSE"): "peeky-reverse",
        ("e07", "SEQD-VAN-REV"): "seq2seq-reverse",
        ("e07", "SEQ2SEQ-REVERSE"): "seq2seq-reverse",
        ("e07", "SEQD-PEEKY-REV"): "peeky-reverse",
        ("e07", "PEEKY-REVERSE"): "peeky-reverse",
        ("e07", "SEQD-ATTN-REV"): "attention-reverse",
        ("e07", "ATTENTION-REVERSE"): "attention-reverse",
    }
    return aliases.get((experiment_id, native), native)


def _declarations(volume, experiment_id):
    if volume is Volume.DS1:
        return (TEST_ACCURACY,)
    return {
        "e03": (TRAIN_PERPLEXITY,),
        "e04": (TEST_PERPLEXITY,),
        "e05": (TEST_PERPLEXITY,),
        "e06": (TEST_EXACT_MATCH,),
        "e07": (TEST_EXACT_MATCH,),
    }.get(experiment_id, ())
