"""W2V1 paper-reproduction training orchestration."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro_io.checksum import sha256_file
from w2v import (
    Corpus,
    Model,
    TrainingConfig,
    TrainingSession,
)

from f2.suites.w2v.artifacts import (
    create_checkpoint_manager,
    resolve_or_build_vocabulary,
    save_lookup_artifact,
)
from f2.suites.w2v.observations import DenseObservationWriter
from repro_core.context import ExperimentContext


@dataclass(frozen=True)
class W2V1Result:
    root: Path
    checkpoint: Path
    lookup: Path
    metrics: Path
    report: Path


class W2V1Executor:
    suite_name = "w2v1"

    def run(self, config: dict[str, object], context: ExperimentContext) -> W2V1Result:
        identity = _mapping(config, "identity")
        corpus_config = _mapping(config, "corpus")
        source = Path(str(corpus_config["path"]))
        if not source.is_absolute():
            source = context.paths.repo_root / source
        if not source.is_file():
            raise ValueError(f"W2V1 corpus does not exist: {source}")
        corpus_digest = _corpus_digest(corpus_config, source)

        run_key = str(identity["planned_run_slot_id"])
        root_override = context.metadata.get("run_root")
        root = (
            Path(root_override)
            if root_override
            else context.paths.run_staging(
                domain="f2",
                suite=self.suite_name,
                study="table2",
                variant="local",
                run_key=run_key,
            )
        )
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

        session = create_session(
            config,
            context.paths.repo_root,
            cache_root=context.paths.cache_root,
            corpus_digest=corpus_digest,
        )
        manager = create_checkpoint_manager(
            root / "checkpoints",
            session=session,
            resource_version=str(identity["resource_version"]),
        )
        writer = DenseObservationWriter(root / "metrics" / "observations.csv")
        reports = []
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
        save_lookup_artifact(
            state, lookup_path, resource_version=str(identity["resource_version"])
        )
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
        return W2V1Result(root, final.path, lookup_path, writer.path, report_path)


def create_session(
    config: dict[str, object],
    repo_root: Path,
    *,
    cache_root: Path | None = None,
    corpus_digest: str | None = None,
) -> TrainingSession:
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
    training = TrainingConfig(**values)
    model = Model.create(vocabulary, training)
    return TrainingSession(
        corpus, vocabulary, model, training, corpus_digest=corpus_digest
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


EXECUTORS = {"word2vec": W2V1Executor()}


def get_executor(kind: str) -> W2V1Executor:
    try:
        return EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown W2V1 experiment kind: {kind}") from exc


__all__ = [
    "W2V1Executor",
    "W2V1Result",
    "create_session",
    "get_executor",
]
