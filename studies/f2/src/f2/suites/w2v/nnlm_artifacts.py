"""Architecture-specific NNLM checkpoint translation; shared W2V lookup boundary."""

from pathlib import Path
from typing import Any

import numpy as np
from w2v import NnlmReplicaState, NnlmTrainingState, VocabularyState

from f2.suites.w2v.artifacts import (
    _VOCABULARY_ARRAYS,
    _load_array,
    _read_json,
    _validate_manifest_identity,
    _write_json,
    _write_manifest,
    verify_artifact,
)
from repro_core.context.checkpoint import CheckpointManager, CheckpointRetentionPolicy

CHECKPOINT_FORMAT = "f2-nnlm-checkpoint-v1"
_PARAMETERS = (
    "input_embeddings.npy",
    "hidden_weights.npy",
    "hidden_bias.npy",
    "output_weights.npy",
)
_ARRAYS = (*_PARAMETERS, *_VOCABULARY_ARRAYS)
_ADAGRAD_ARRAYS = (
    "input_adagrad.npy",
    "hidden_adagrad.npy",
    "bias_adagrad.npy",
    "output_adagrad.npy",
)


def save_checkpoint(
    state: NnlmTrainingState, path: Path, *, payload: str = "full"
) -> None:
    if payload != "full":
        raise ValueError("NNLM checkpoints require a full resumable payload")
    path.mkdir(parents=True, exist_ok=False)
    vocabulary = state.vocabulary_state()
    for name, getter in zip(
        _PARAMETERS,
        (
            state.input_embeddings,
            state.hidden_weights,
            state.hidden_bias,
            state.output_weights,
        ),
        strict=True,
    ):
        value = getter()
        np.save(path / name, value, allow_pickle=False)
        del value
    for name, value in zip(_VOCABULARY_ARRAYS, vocabulary.arrays(), strict=True):
        np.save(path / name, value, allow_pickle=False)
    array_names = _ARRAYS
    worker_state: dict[str, Any] = {"boundary": "epoch", "stochastic_training": False}
    if state.optimizer_kind == "adagrad":
        array_names += _ADAGRAD_ARRAYS
        for name, value in zip(_ADAGRAD_ARRAYS, state.adagrad_arrays(), strict=True):
            np.save(path / name, value, allow_pickle=False)
        worker_state["replicas"] = [
            {
                key: getattr(replica, key)
                for key in (
                    "replica_id",
                    "start_byte",
                    "end_byte",
                    "processed_tokens",
                    "objective_count",
                    "batch_count",
                )
            }
            for replica in state.replicas()
        ]
    trainer = {
        "schema_version": state.schema_version,
        "config_digest": state.config_digest,
        "vocabulary_digest": state.vocabulary_digest,
        "corpus_digest": state.corpus_digest,
        "completed_epochs": state.completed_epochs,
        "processed_tokens": state.processed_tokens,
        "history_length": state.history_length,
        "vocabulary": {
            "hash_capacity": vocabulary.hash_capacity,
            "retained_token_count": vocabulary.retained_token_count,
        },
        # Neither NNLM corpus scan consumes RNG during training.
        # Checkpoints are supported only between complete epochs: no pending
        # history, cache, gradient batch, tokenizer cursor, or queued PS update
        # survives an epoch boundary. Downpour joins workers and drains all PS
        # queues before allowing an export.
        "optimizer": state.optimizer_kind,
        "worker_state": worker_state,
    }
    _write_json(path / "trainer_state.json", trainer)
    _write_manifest(
        path, CHECKPOINT_FORMAT, trainer, (*array_names, "trainer_state.json")
    )


def load_checkpoint(
    path: Path, *, expected: dict[str, str] | None = None
) -> NnlmTrainingState:
    manifest = verify_artifact(path, expected=expected, format_name=CHECKPOINT_FORMAT)
    trainer = _read_json(path / "trainer_state.json")
    _validate_manifest_identity(manifest, trainer)
    if trainer != {key: manifest[key] for key in trainer}:
        raise ValueError("NNLM manifest and trainer state differ")
    optimizer = trainer["optimizer"]
    workers = trainer["worker_state"]
    if (
        optimizer not in ("sgd", "adagrad")
        or workers.get("boundary") != "epoch"
        or workers.get("stochastic_training") is not False
    ):
        raise ValueError("unsupported NNLM worker or optimizer state")
    names = _ARRAYS + (_ADAGRAD_ARRAYS if optimizer == "adagrad" else ())
    if set(manifest["files"]) != {*names, "trainer_state.json"}:
        raise ValueError("invalid NNLM checkpoint file set")
    if set(workers) != (
        {"boundary", "stochastic_training", "replicas"}
        if optimizer == "adagrad"
        else {"boundary", "stochastic_training"}
    ):
        raise ValueError("invalid NNLM worker state")
    arrays = {name: _load_array(path / name) for name in names}
    vocabulary = VocabularyState.from_arrays(
        *(arrays[name] for name in _VOCABULARY_ARRAYS),
        int(trainer["vocabulary"]["hash_capacity"]),
        int(trainer["vocabulary"]["retained_token_count"]),
    )
    state = NnlmTrainingState.from_parts(
        int(trainer["schema_version"]),
        str(trainer["config_digest"]),
        str(trainer["vocabulary_digest"]),
        str(trainer["corpus_digest"]),
        int(trainer["completed_epochs"]),
        int(trainer["processed_tokens"]),
        int(trainer["history_length"]),
        vocabulary,
        *(arrays[name] for name in _PARAMETERS),
    )
    if optimizer == "adagrad":
        replicas = [NnlmReplicaState(**record) for record in workers["replicas"]]
        if (
            not replicas
            or sum(replica.processed_tokens for replica in replicas)
            != state.processed_tokens
        ):
            raise ValueError("invalid NNLM replica token counters")
        return NnlmTrainingState.from_downpour_parts(
            state, *(arrays[name] for name in _ADAGRAD_ARRAYS), replicas
        )
    return state


def create_checkpoint_manager(
    root: Path, *, session: Any, policy: CheckpointRetentionPolicy | None = None
) -> CheckpointManager:
    policy = policy or CheckpointRetentionPolicy()
    if policy.best_payload != "full":
        raise ValueError("NNLM best checkpoints require full state")
    return CheckpointManager(
        root=root,
        config_digest=session.config_digest,
        save_fn=lambda path, payload: save_checkpoint(
            session.export_state(), path, payload=payload
        ),
        epoch_fn=lambda: session.completed_epochs,
        step_fn=lambda: session.processed_tokens,
        policy=policy,
    )
