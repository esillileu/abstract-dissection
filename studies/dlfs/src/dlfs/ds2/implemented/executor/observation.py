from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from deepscratch.core import Tensor, configure_runtime
from deepscratch.nn.model.architecture import AttentionSeq2seq

from dlfs.adapters.checkpoint import load_deepscratch_model_parameters
from dlfs.ds2.implemented import executor
from dlfs.ds2.implemented.adapters import build_seq2seq_model
from repro_core.context import ExperimentContext
from repro_core.context.checkpoint import resolve_checkpoint_path
from repro_core.context.contracts import ExperimentResult

from ..records import DS2Records
from .attention_weights import (
    _attention_example_ids,
    _generate_attention_with_weights,
    _teacher_forced_attention_with_weights,
)
from .common import _artifact_root, _mapping
from .digest import (
    _array_digest,
    _file_digest,
    _path_digest,
    _sequence_dataset_path,
)
from .seq_predictions import _decode_ids


def get_observation_executor(config: dict[str, object]):
    group_id = str(config.get("execution_group_id", ""))
    if group_id == "GO01":
        return AttentionAlignmentObservationExecutor()
    raise ValueError(f"unknown DS2 observation group: {group_id}")


class AttentionAlignmentObservationExecutor:
    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset = _mapping(config, "dataset")
        model_config = _mapping(config, "model")
        checkpoint_config = _mapping(config, "checkpoint")
        checkpoint_path = checkpoint_config.get("source_path") or checkpoint_config.get(
            "source_checkpoint_path"
        )
        if checkpoint_path is None:
            raise ValueError("DS2 GO01 requires checkpoint.source_path")
        checkpoint_path = resolve_checkpoint_path(Path(str(checkpoint_path)))
        data = executor.load_ds2_sequence(
            str(dataset["file"]),
            seed=int(dataset.get("split_seed", 1984)),
            split_algorithm=str(
                dataset.get("split_algorithm", "legacy_numpy_randomstate")
            ),
        )
        x_test, t_test = data["test"]
        if bool(dataset.get("reverse", True)):
            x_test = x_test[:, ::-1]
        context.metadata["data"] = {
            "split_seed": int(dataset.get("split_seed", 1984)),
            "split_algorithm": str(
                dataset.get("split_algorithm", "legacy_numpy_randomstate")
            ),
            "dataset_checksum": _file_digest(
                _sequence_dataset_path(str(dataset["file"]))
            ),
            "observation_split_checksum": _array_digest(x_test, t_test),
        }
        model = build_seq2seq_model(
            str(model_config.get("name", "AttentionSeq2seq")),
            len(data["char_to_id"]),
            model_config,
            backend,
        )
        if not isinstance(model, AttentionSeq2seq):
            raise ValueError("DS2 GO01 requires AttentionSeq2seq model")
        load_deepscratch_model_parameters(Path(str(checkpoint_path)), model)
        artifact_root = _artifact_root(config, context)
        records = DS2Records()
        records.bind_artifact_root(artifact_root)
        recording = _mapping(config, "recording")
        attention_config = _mapping(recording, "attention")
        count = int(attention_config.get("count", 5))
        example_ids = _attention_example_ids(
            size=len(x_test),
            count=count,
            seed=int(attention_config.get("selection_seed", 1984)),
        )
        start_id = data["char_to_id"]["_"]
        id_to_char = data["id_to_char"]
        render_examples = []
        for example_id in example_ids:
            question = Tensor(
                backend.xp.asarray(
                    x_test[example_id : example_id + 1], dtype=backend.xp.int64
                ),
                backend=backend,
            )
            expected = [int(value) for value in t_test[example_id][1:]]
            conditioning = str(
                attention_config.get(
                    "conditioning", attention_config.get("decode", "greedy")
                )
            )
            if conditioning == "teacher_forcing":
                predicted, weights = _teacher_forced_attention_with_weights(
                    model,
                    question,
                    t_test[example_id],
                    backend,
                )
            elif conditioning == "greedy":
                predicted, weights = _generate_attention_with_weights(
                    model, question, start_id, len(expected), backend
                )
            else:
                raise ValueError(
                    "attention.conditioning must be teacher_forcing or greedy"
                )
            weights = weights[:, ::-1]
            source_text = _decode_ids(x_test[example_id], id_to_char)
            target_text = _decode_ids(expected, id_to_char)
            prediction_text = _decode_ids(predicted, id_to_char)
            records.add_prediction(
                {
                    "epoch": 0,
                    "example_id": example_id,
                    "source": source_text,
                    "target": target_text,
                    "prediction": prediction_text,
                    "exact_match": int(predicted == expected),
                    "token_correct": sum(
                        left == right
                        for left, right in zip(predicted, expected, strict=True)
                    ),
                    "token_count": len(expected),
                }
            )
            render_examples.append(
                {
                    "example_id": example_id,
                    "source": source_text,
                    "target": target_text,
                    "prediction": prediction_text,
                    "source_labels": list(source_text),
                    "target_labels": list(target_text),
                }
            )
            for decode_step in range(weights.shape[0]):
                for encoder_position in range(weights.shape[1]):
                    records.add_attention(
                        {
                            "example_id": example_id,
                            "decode_step": decode_step,
                            "encoder_position": encoder_position,
                            "weight": float(weights[decode_step, encoder_position]),
                        }
                    )
        records.set_attention_render(
            {
                "source_checkpoint": str(Path(str(checkpoint_path))),
                "source_checkpoint_sha256": _path_digest(Path(str(checkpoint_path))),
                "example_selection_seed": int(
                    attention_config.get("selection_seed", 1984)
                ),
                "decode_policy": str(attention_config.get("decode", "greedy")),
                "conditioning": str(
                    attention_config.get(
                        "conditioning", attention_config.get("decode", "greedy")
                    )
                ),
                "condition": str(config.get("atomic_run_id", "")),
                "input_reversal": bool(dataset.get("reverse", True)),
                "x_axis": "encoder_position",
                "y_axis": "decode_step",
                "y_axis_inverted": True,
                "color_range": [0.0, 1.0],
                "examples": render_examples,
            }
        )
        records.flush()
        return ExperimentResult(
            metrics={
                "final/status/success": 1.0,
                "final/system/total_updates": 0.0,
                "final/system/completed_epochs": 0.0,
                "final/system/samples_seen": float(len(example_ids)),
            },
            artifact_root=artifact_root,
            model=model,
            metric_rows=(),
            profiling_metrics={},
        )
