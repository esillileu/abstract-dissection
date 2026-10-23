"""Word2Vec paper-reproduction base training orchestration and session factory."""

from __future__ import annotations

import json
import shutil
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

from repro_core.context import ExperimentContext

from .artifacts import (
    create_checkpoint_manager,
    resolve_or_build_vocabulary,
    save_lookup_artifact,
)
from .nnlm_artifacts import (
    create_checkpoint_manager as create_nnlm_checkpoint_manager,
)
from .observations import DenseObservationWriter


@dataclass(frozen=True)
class Word2VecResult:
    """Artifact bundle output from completed Word2Vec execution."""

    root: Path
    checkpoint: Path
    lookup: Path
    metrics: Path
    report: Path


class Word2VecExecutor:
    """Base orchestration executor for Word2Vec training runs."""

    suite_name: str = "w2v"

    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> Word2VecResult:
        identity = _mapping(config, "identity")
        corpus_config = _mapping(config, "corpus")
        source = Path(str(corpus_config["path"]))
        if not source.is_absolute():
            source = context.paths.repo_root / source
        if not source.is_file():
            raise ValueError(
                f"{self.suite_name.upper()} corpus does not exist: {source}"
            )
        corpus_digest = _corpus_digest(corpus_config, source)

        run_key = str(identity["planned_run_slot_id"])
        root_override = context.metadata.get("run_root")
        root = (
            Path(root_override)
            if root_override
            else context.paths.run_staging(
                domain="f2",
                suite=self.suite_name,
                study=str(identity.get("study", "table2")),
                variant="local",
                run_key=run_key,
            )
        )
        checkpoint_dir = root / "checkpoints"
        latest_json = checkpoint_dir / "latest.json"
        resumed_session = None
        prior_reports: list[dict[str, Any]] = []

        if latest_json.is_file():
            try:
                resumed_session = restore_session(
                    config,
                    context.paths.repo_root,
                    latest_json,
                    cache_root=context.paths.cache_root,
                    corpus_digest=corpus_digest,
                )
                obs_file = root / "metrics" / "observations.csv"
                if obs_file.is_file():
                    import csv

                    prior_by_epoch: dict[int, dict[str, Any]] = {}
                    with obs_file.open("r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            ep = int(row["epoch"])
                            prior_by_epoch[ep] = {
                                "epoch": ep,
                                "processed_tokens": int(row["processed_tokens"]),
                                "objective_loss": float(row["objective_loss"]),
                            }
                    for ep in sorted(prior_by_epoch):
                        if ep <= resumed_session.completed_epochs:
                            prior_reports.append(prior_by_epoch[ep])
            except Exception:
                resumed_session = None

        if resumed_session is not None:
            session = resumed_session
            reports = prior_reports
        else:
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)
            session = create_session(
                config,
                context.paths.repo_root,
                cache_root=context.paths.cache_root,
                corpus_digest=corpus_digest,
            )
            reports = []

        checkpoint_factory = (
            create_nnlm_checkpoint_manager
            if _mapping(config, "training")["model_kind"] == "feedforward_nnlm"
            else create_checkpoint_manager
        )
        manager = checkpoint_factory(
            checkpoint_dir,
            session=session,
        )
        writer = DenseObservationWriter(root / "metrics" / "observations.csv")
        progress = context.metadata.get("progress_reporter")
        total_epochs = int(_mapping(config, "training")["epochs"])
        if progress is not None:
            progress.set_total_updates(total_epochs, completed=session.completed_epochs)
            progress.write(
                f"preparing {self.suite_name} slot={run_key} "
                f"epochs={total_epochs} completed={session.completed_epochs}"
            )
        while not session.is_complete:
            epoch = session.train_epoch()
            writer.append(epoch.observations())
            manager.save_latest()
            reports.append(
                {
                    "epoch": epoch.epoch,
                    "processed_tokens": epoch.processed_tokens,
                    "objective_loss": epoch.objective_loss,
                }
            )
            if progress is not None:
                progress.advance_to(epoch.epoch, epoch=epoch.epoch)
                progress.write(
                    f"{self.suite_name} epoch={epoch.epoch}/{total_epochs} "
                    f"tokens={epoch.processed_tokens} "
                    f"loss={epoch.objective_loss} "
                    f"tokens_per_second={epoch.tokens_per_second:.1f}"
                )
        if progress is not None:
            progress.write(f"publishing {self.suite_name} artifacts slot={run_key}")
        final = manager.save_final()
        state = session.export_state()
        lookup_path = root / "lookup"
        # Lookup is derived from the restored final state and may already exist
        # when a completed run is invoked again or publication is retried.
        if lookup_path.exists():
            shutil.rmtree(lookup_path)
        save_lookup_artifact(state, lookup_path)
        report = {
            "identity": identity,
            "epochs": reports,
            "complete": session.is_complete,
            "artifacts": {
                "checkpoint": str(final.path.relative_to(root)),
                "lookup": str(lookup_path.relative_to(root)),
                "metrics": "metrics/observations.csv",
            },
        }
        report_path = root / "result.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return Word2VecResult(root, final.path, lookup_path, writer.path, report_path)


def create_session(
    config: dict[str, object],
    repo_root: Path,
    *,
    cache_root: Path | None = None,
    corpus_digest: str | None = None,
) -> (
    TrainingSession
    | DownpourTrainingSession
    | NnlmTrainingSession
    | NnlmDownpourTrainingSession
):
    corpus_path = Path(str(_mapping(config, "corpus")["path"]))
    corpus = Corpus(
        corpus_path if corpus_path.is_absolute() else repo_root / corpus_path
    )
    corpus_digest = corpus_digest or _corpus_digest(
        _mapping(config, "corpus"), Path(corpus.path)
    )
    vocabulary = resolve_or_build_vocabulary(
        corpus,
        _mapping(config, "vocabulary"),
        cache_root=cache_root or repo_root / ".cache",
        corpus_digest=corpus_digest,
    )
    values = dict(_mapping(config, "training"))
    values["root_seed"] = int(_mapping(config, "identity")["seed"])
    if values.get("model_kind") == "feedforward_nnlm":
        values.pop("model_kind")
        training = NnlmTrainingConfig(**values)
        distribution = config.get("distribution")
        if distribution is not None:
            distribution = _mapping(config, "distribution")
            mode = distribution.pop("mode", None)
            if mode == "downpour":
                return NnlmDownpourTrainingSession(
                    corpus,
                    vocabulary,
                    training,
                    DownpourConfig(**distribution),
                    corpus_digest=corpus_digest,
                )
            if mode != "none":
                raise ValueError(f"unsupported NNLM distribution mode: {mode}")
        return NnlmTrainingSession(
            corpus, vocabulary, training, corpus_digest=corpus_digest
        )
    training = TrainingConfig(**values)
    model = Model.create(vocabulary, training)

    distribution = config.get("distribution")
    if distribution is not None and isinstance(distribution, dict):
        dist_dict = dict(distribution)
        mode = dist_dict.get("mode")
        if mode == "downpour":
            downpour_kwargs: dict[str, Any] = {}
            for field in (
                "parameter_server_shards",
                "mini_batch_targets",
                "adagrad_gamma",
                "adagrad_epsilon",
                "queue_capacity",
                "fetch_interval",
                "push_interval",
            ):
                if field in dist_dict:
                    downpour_kwargs[field] = dist_dict[field]
            downpour_cfg = DownpourConfig(**downpour_kwargs)
            return DownpourTrainingSession(
                corpus,
                vocabulary,
                model,
                training,
                downpour_cfg,
                corpus_digest=corpus_digest,
            )
        if mode != "none":
            raise ValueError(f"unsupported distribution mode: {mode}")

    return TrainingSession(
        corpus, vocabulary, model, training, corpus_digest=corpus_digest
    )


def restore_session(
    config: dict[str, object],
    repo_root: Path,
    latest_json_path: Path,
    *,
    cache_root: Path | None = None,
    corpus_digest: str | None = None,
) -> (
    TrainingSession
    | DownpourTrainingSession
    | NnlmTrainingSession
    | NnlmDownpourTrainingSession
):
    checkpoint_dir = latest_json_path.parent
    latest_meta = json.loads(latest_json_path.read_text())
    ckpt_dir = checkpoint_dir / str(latest_meta["path"])

    corpus_path = Path(str(_mapping(config, "corpus")["path"]))
    corpus = Corpus(
        corpus_path if corpus_path.is_absolute() else repo_root / corpus_path
    )
    corpus_digest = corpus_digest or _corpus_digest(
        _mapping(config, "corpus"), Path(corpus.path)
    )
    vocabulary = resolve_or_build_vocabulary(
        corpus,
        _mapping(config, "vocabulary"),
        cache_root=cache_root or repo_root / ".cache",
        corpus_digest=corpus_digest,
    )
    values = dict(_mapping(config, "training"))
    values["root_seed"] = int(_mapping(config, "identity")["seed"])

    if values.get("model_kind") == "feedforward_nnlm":
        from .nnlm_artifacts import load_checkpoint as load_nnlm_checkpoint

        state = load_nnlm_checkpoint(ckpt_dir)
        values.pop("model_kind")
        training = NnlmTrainingConfig(**values)
        distribution = config.get("distribution")
        if distribution is not None:
            distribution = _mapping(config, "distribution")
            mode = distribution.pop("mode", None)
            if mode == "downpour":
                return NnlmDownpourTrainingSession.restore(
                    corpus,
                    vocabulary,
                    training,
                    DownpourConfig(**distribution),
                    state,
                    corpus_digest=corpus_digest,
                )
            if mode != "none":
                raise ValueError(f"unsupported NNLM distribution mode: {mode}")
        return NnlmTrainingSession.restore(
            corpus, vocabulary, training, state, corpus_digest=corpus_digest
        )

    from .artifacts import load_checkpoint as load_w2v_checkpoint

    w2v_state = load_w2v_checkpoint(ckpt_dir)
    training = TrainingConfig(**values)
    model = Model.create(vocabulary, training)
    return TrainingSession.restore(
        corpus, vocabulary, model, training, w2v_state, corpus_digest=corpus_digest
    )


def _corpus_digest(corpus_config: dict[str, Any], source: Path) -> str:
    """Use a materializer-verified identity, hashing only unbound local inputs."""
    configured = corpus_config.get("sha256")
    return str(configured) if configured is not None else sha256_file(source)


def _mapping(config: dict[str, object], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


__all__ = [
    "Word2VecExecutor",
    "Word2VecResult",
    "_corpus_digest",
    "_mapping",
    "create_session",
]
