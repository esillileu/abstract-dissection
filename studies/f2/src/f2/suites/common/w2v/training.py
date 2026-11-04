"""Word2Vec training session factory and execution routines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro_io.checksum import sha256_file

from w2v import (
    Corpus,
    DownpourConfig,
    DownpourTrainingSession,
    Model,
    NnlmDownpourTrainingSession,
    NnlmTrainingConfig,
    NnlmTrainingSession,
    TrainingConfig,
    TrainingSession,
)

from .checkpoints import load_checkpoint
from .preparation import resolve_or_build_vocabulary


@dataclass(frozen=True)
class Word2VecResult:
    """Artifact bundle output from completed Word2Vec execution."""

    root: Path
    checkpoint: Path
    lookup: Path
    metrics: Path
    report: Path


@dataclass(frozen=True)
class W2VRunReceipt:
    result: object
    run_id: str
    staging_root: Path
    skipped: bool = False


def corpus_digest(corpus_config: dict[str, Any], source: Path) -> str:
    """Use a materializer-verified identity, hashing only unbound local inputs."""
    cfg = corpus_config.get("sha256")
    return str(cfg) if cfg is not None else sha256_file(source)


def mapping(config: dict[str, object], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def create_session(
    config: dict[str, object],
    repo_root: Path,
    *,
    cache_root: Path | None = None,
    corpus_digest: str | None = None,
    resume_from: Path | None = None,
) -> (
    TrainingSession
    | DownpourTrainingSession
    | NnlmTrainingSession
    | NnlmDownpourTrainingSession
):
    corpus_path = Path(str(mapping(config, "corpus")["path"]))
    corpus = Corpus(
        corpus_path if corpus_path.is_absolute() else repo_root / corpus_path,
        full_tokens="phrase_detection" in config,
    )
    c_digest = corpus_digest or _corpus_digest_fallback(
        mapping(config, "corpus"), Path(corpus.path)
    )
    vocabulary = resolve_or_build_vocabulary(
        corpus,
        mapping(config, "vocabulary"),
        cache_root=cache_root or repo_root / "workspace/inputs/f2/prepared",
        corpus_digest=c_digest,
    )
    vals = dict(mapping(config, "training"))
    vals["root_seed"] = int(mapping(config, "identity")["seed"])
    if vals.get("model_kind") == "feedforward_nnlm":
        vals.pop("model_kind")
        nnlm_cfg = NnlmTrainingConfig(**vals)
        dist = config.get("distribution")
        if dist is not None:
            dist_map = mapping(config, "distribution")
            mode = dist_map.pop("mode", None)
            if mode == "downpour":
                return NnlmDownpourTrainingSession(
                    corpus,
                    vocabulary,
                    nnlm_cfg,
                    DownpourConfig(**dist_map),
                    corpus_digest=c_digest,
                )
            if mode != "none":
                raise ValueError(f"unsupported NNLM distribution: {dist_map}")
        return NnlmTrainingSession(corpus, vocabulary, nnlm_cfg, corpus_digest=c_digest)
    training = TrainingConfig(**vals)
    model = Model.create(vocabulary, training)

    if resume_from is not None:
        state = load_checkpoint(resume_from)
        model.restore_input_embeddings(state.input_embeddings())
        model.restore_output_embeddings(state.output_embeddings())
        return TrainingSession.restore(
            corpus, vocabulary, model, training, state, corpus_digest=c_digest
        )

    dist = config.get("distribution")
    if dist is not None and isinstance(dist, dict):
        dist_dict = dict(dist)
        if dist_dict.get("mode") == "downpour":
            dp_keys = (
                "parameter_server_shards",
                "mini_batch_targets",
                "adagrad_gamma",
                "adagrad_epsilon",
                "queue_capacity",
                "fetch_interval",
                "push_interval",
            )
            dp_kwargs = {k: dist_dict[k] for k in dp_keys if k in dist_dict}
            return DownpourTrainingSession(
                corpus,
                vocabulary,
                model,
                training,
                DownpourConfig(**dp_kwargs),
                corpus_digest=c_digest,
            )
        if dist_dict.get("mode") != "none":
            raise ValueError(f"unsupported distribution mode: {dist_dict.get('mode')}")

    return TrainingSession(corpus, vocabulary, model, training, corpus_digest=c_digest)


def _corpus_digest_fallback(corpus_config: dict[str, Any], source: Path) -> str:
    configured = corpus_config.get("sha256")
    return str(configured) if configured is not None else sha256_file(source)


__all__ = [
    "W2VRunReceipt",
    "Word2VecResult",
    "corpus_digest",
    "create_session",
    "mapping",
]
