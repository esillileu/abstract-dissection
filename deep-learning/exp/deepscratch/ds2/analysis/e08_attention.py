"""DS2 GO01: reproduce one attention-alignment heatmap per fixed example."""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import Curve, save_figure, write_summary
from exp.deepscratch.analysis.input import artifact_file, artifact_rows

from .common import runs


def _condition_slug(condition: str) -> str:
    return condition.lower().replace("-", "_")


def _matrices(client, run_refs):
    by_example = defaultdict(list)
    for run in run_refs:
        cells = defaultdict(dict)
        for row in artifact_rows(client, run, "observations/attention.csv"):
            try:
                cells[row["example_id"]][(int(row["decode_step"]), int(row["encoder_position"]))] = float(
                    row["weight"]
                )
            except (KeyError, TypeError, ValueError):
                continue
        for example, values in cells.items():
            if values:
                by_example[example].append(values)
    result = {}
    for example, histories in by_example.items():
        common = set(histories[0])
        for history in histories[1:]:
            common.intersection_update(history)
        if not common:
            continue
        row_count = max(cell[0] for cell in common) + 1
        column_count = max(cell[1] for cell in common) + 1
        stack = []
        for history in histories:
            matrix = np.full((row_count, column_count), np.nan)
            for cell in common:
                matrix[cell] = history[cell]
            stack.append(matrix)
        values = np.asarray(stack)
        result[example] = (
            np.nanmean(values, axis=0),
            np.nanmin(values, axis=0),
            np.nanmax(values, axis=0),
            len(values),
        )
    return result


def _labels(client, run_refs):
    if not run_refs:
        return {}
    path = artifact_file(client, run_refs[0], "observations/attention_render.json")
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    examples = value.get("examples", value)
    if isinstance(examples, dict):
        labels = examples
    elif isinstance(examples, list):
        labels = {
            str(example["example_id"]): example
            for example in examples
            if isinstance(example, dict) and "example_id" in example
        }
    else:
        return {}
    if value.get("input_reversal") is not True:
        return labels
    return {
        example_id: {
            **metadata,
            "source_labels": list(reversed(metadata.get("source_labels", []))),
        }
        for example_id, metadata in labels.items()
        if isinstance(metadata, dict)
    }


def render(client, error_style, output):
    del error_style  # The source output is a heatmap, not a scalar error-axis graph.
    conditions = ["ATTENTION-ALIGNMENT", "ATTENTION-ALIGNMENT-GREEDY"]
    grouped = runs(client, "GO01", conditions)
    outputs = []
    for condition in conditions:
        run_refs = grouped[condition]
        matrices = _matrices(client, run_refs)
        labels = _labels(client, run_refs)
        condition_slug = _condition_slug(condition)
        for example, (mean, minimum, maximum, count) in sorted(matrices.items()):
            example_output = output.with_name(
                f"{output.stem}_{condition_slug}_{example}{output.suffix}"
            )
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.pcolor(mean, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)
            metadata = labels.get(example, {}) if isinstance(labels, dict) else {}
            source_labels = metadata.get("source_labels") if isinstance(metadata, dict) else None
            target_labels = metadata.get("target_labels") if isinstance(metadata, dict) else None
            if source_labels:
                axis.set_xticks(np.arange(len(source_labels)) + 0.5, source_labels)
            if target_labels:
                axis.set_yticks(np.arange(len(target_labels)) + 0.5, target_labels)
            axis.invert_yaxis()
            axis.set(xlabel="encoder character position", ylabel="decoder character position")
            save_figure(figure, example_output)
            plt.close(figure)
            outputs.append(example_output)
        summary = output.with_name(f"{output.stem}_{condition_slug}.csv")
        write_summary(
            summary,
            {
                example: Curve(np.arange(mean.size), mean.ravel(), minimum.ravel(), maximum.ravel(), count)
                for example, (mean, minimum, maximum, count) in matrices.items()
            },
        )
        outputs.append(summary)
    return outputs
