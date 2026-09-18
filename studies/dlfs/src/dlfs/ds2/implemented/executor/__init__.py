"""Portable PTB word2vec/RNNLM and character seq2seq experiment executors."""

from __future__ import annotations

from dlfs.ds2.implemented.adapters import load_ds2_ptb, load_ds2_sequence

from .attention_weights import (
    _attention_example_ids,
    _generate_attention_with_weights,
    _teacher_forced_attention_with_weights,
)
from .checkpoint import (
    _config_digest,
    _publish_array_checkpoint,
    _record_retained_checkpoints,
    _save_epoch_roles,
)
from .common import (
    _apply_validation_decay,
    _artifact_root,
    _backend_exp_float,
    _device_timer,
    _final,
    _mapping,
    _recorded_float,
    _source_curve_from_objective,
)
from .count_based import CountBasedEmbeddingExecutor
from .digest import (
    _array_digest,
    _file_digest,
    _path_digest,
    _sequence_dataset_path,
)
from .language_model import LanguageModelExecutor
from .observation import (
    AttentionAlignmentObservationExecutor,
    get_observation_executor,
)
from .profile import ProfileExecutor
from .seq2seq import Seq2SeqExecutor
from .seq_predictions import (
    _decode_ids,
    _record_seq_predictions,
)
from .word2vec import Word2VecExecutor

_EXECUTORS = {
    "count_based_embedding": CountBasedEmbeddingExecutor(),
    "performance_profile": ProfileExecutor(),
    "word2vec": Word2VecExecutor(),
    "language_modeling": LanguageModelExecutor(),
    "seq2seq": Seq2SeqExecutor(),
}


def get_executor(kind: str):
    try:
        return _EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown DS2 implemented experiment kind: {kind}") from exc


__all__ = [
    "AttentionAlignmentObservationExecutor",
    "CountBasedEmbeddingExecutor",
    "LanguageModelExecutor",
    "ProfileExecutor",
    "Seq2SeqExecutor",
    "Word2VecExecutor",
    "_apply_validation_decay",
    "_array_digest",
    "_artifact_root",
    "_attention_example_ids",
    "_backend_exp_float",
    "_config_digest",
    "_decode_ids",
    "_device_timer",
    "_file_digest",
    "_final",
    "_generate_attention_with_weights",
    "_mapping",
    "_path_digest",
    "_publish_array_checkpoint",
    "_record_retained_checkpoints",
    "_record_seq_predictions",
    "_recorded_float",
    "_save_epoch_roles",
    "_sequence_dataset_path",
    "_source_curve_from_objective",
    "_teacher_forced_attention_with_weights",
    "get_executor",
    "get_observation_executor",
    "load_ds2_ptb",
    "load_ds2_sequence",
]
