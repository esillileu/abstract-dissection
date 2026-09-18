"""Local epoch-boundary W2V1 orchestration."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from w2v import (
    Corpus,
    Model,
    TrainingConfig,
    TrainingSession,
    Vocabulary,
    VocabularyConfig,
)

from f2.suites.w2v.artifacts import (
    create_checkpoint_manager,
    load_checkpoint,
    load_lookup_artifact,
    save_lookup_artifact,
)
from f2.suites.w2v.evaluation import evaluate_analogies, parse_analogy_questions
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
        import hashlib

        actual_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        expected_file_digest = str(
            corpus_config.get("sha256", identity["corpus_manifest_digest"])
        )
        if actual_digest != expected_file_digest:
            raise ValueError("W2V1 corpus identity does not match the resolved config")

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

        resume_checkpoint = context.metadata.get("resume_checkpoint")
        session = (
            restore_session(config, Path(resume_checkpoint), context.paths.repo_root)
            if resume_checkpoint
            else create_session(config, context.paths.repo_root)
        )
        manager = create_checkpoint_manager(
            root / "checkpoints",
            session=session,
            resource_version=str(identity["resource_version"]),
        )
        writer = DenseObservationWriter(root / "metrics" / "observations.csv")
        reports = []
        stop_after_epoch = context.metadata.get("stop_after_epoch")
        while not session.is_complete and (
            stop_after_epoch is None or session.completed_epochs < int(stop_after_epoch)
        ):
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
        final = manager.save_final() if session.is_complete else manager.save_latest()
        state = session.export_state()
        lookup_path = root / "lookup"
        save_lookup_artifact(
            state, lookup_path, resource_version=str(identity["resource_version"])
        )
        lookup = load_lookup_artifact(lookup_path)
        questions_path = Path(str(_mapping(config, "evaluation")["questions_path"]))
        if not questions_path.is_absolute():
            questions_path = context.paths.repo_root / questions_path
        evaluation = evaluate_analogies(
            lookup, parse_analogy_questions(questions_path.read_bytes().splitlines())
        ).overall
        report = {
            "identity": identity,
            "epochs": reports,
            "evaluation": asdict(evaluation),
            "coverage": evaluation.coverage,
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


def restore_session(
    config: dict[str, object], checkpoint: Path, repo_root: Path
) -> TrainingSession:
    """Reconstruct a session through the same identity-checked adapter path."""
    corpus_path = Path(str(_mapping(config, "corpus")["path"]))
    corpus = Corpus(
        corpus_path if corpus_path.is_absolute() else repo_root / corpus_path
    )
    vocabulary = Vocabulary.build(
        corpus, VocabularyConfig(**_mapping(config, "vocabulary"))
    )
    values = dict(_mapping(config, "training"))
    values["root_seed"] = int(_mapping(config, "identity")["seed"])
    training = TrainingConfig(**values)
    model = Model.create(vocabulary, training)
    return TrainingSession.restore(
        corpus, vocabulary, model, training, load_checkpoint(checkpoint)
    )


def create_session(config: dict[str, object], repo_root: Path) -> TrainingSession:
    corpus_path = Path(str(_mapping(config, "corpus")["path"]))
    corpus = Corpus(
        corpus_path if corpus_path.is_absolute() else repo_root / corpus_path
    )
    vocabulary = Vocabulary.build(
        corpus, VocabularyConfig(**_mapping(config, "vocabulary"))
    )
    values = dict(_mapping(config, "training"))
    values["root_seed"] = int(_mapping(config, "identity")["seed"])
    training = TrainingConfig(**values)
    model = Model.create(vocabulary, training)
    return TrainingSession(corpus, vocabulary, model, training)


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
    "restore_session",
]
