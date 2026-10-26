"""Word2Vec paper-reproduction base training orchestration and session factory."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
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
from repro_core.execution.locator import PathRunLocator
from repro_core.storage.local_store import LocalStore

from .artifacts import (
    create_checkpoint_manager,
    load_checkpoint,
    lookup_from_checkpoint,
    resolve_or_build_vocabulary,
    save_lookup_artifact,
    vocabulary_cache_path,
)
from .engine import Word2VecEngine
from .nnlm_artifacts import (
    create_checkpoint_manager as create_nnlm_checkpoint_manager,
)
from .observations import DenseObservationWriter
from .tracking import find_matching_finished_run, publish_run_to_mlflow


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


class Word2VecExecutor:
    """Orchestration executor for Word2Vec training runs."""

    suite_name: str = "w2v"

    def execute(
        self,
        config: dict[str, object],
        context: ExperimentContext,
        *,
        tracking_uri: str,
        override: bool = False,
        progress_reporter: Any | None = None,
    ) -> W2VRunReceipt:
        identity = _mapping(config, "identity")
        slot_id = str(identity["planned_run_slot_id"])
        config_digest = str(identity["config_digest"])
        seed = int(identity["seed"])
        experiment_name = str(
            _mapping(config, "tracking").get("experiment", f"f2.{self.suite_name}")
        )

        root_override = context.metadata.get("run_root")
        run_root = (
            Path(root_override)
            if root_override
            else context.paths.runs("f2", self.suite_name, slot_id)
        )
        if override and run_root.exists() and any(run_root.iterdir()):
            idx = 1
            while (run_root / f"run_{idx:02d}").exists() and any(
                (run_root / f"run_{idx:02d}").iterdir()
            ):
                idx += 1
            run_root = run_root / f"run_{idx:02d}"

        local_store = LocalStore(run_root)
        local_store.ensure_directories()
        client = MlflowClient(tracking_uri=tracking_uri)

        # 1. Skip check: MLflow FINISHED run (if not already verified before corpus materialization)
        if not override and not context.metadata.get("skip_checked"):
            existing = find_matching_finished_run(
                client, experiment_name, config_digest, seed
            )
            if existing is not None:
                if progress_reporter is not None:
                    progress_reporter.write(
                        f"Skipping {self.suite_name} slot={slot_id}: finished run {existing.info.run_id} exists"
                    )
                return W2VRunReceipt(
                    result=None,
                    run_id=existing.info.run_id,
                    staging_root=run_root,
                    skipped=True,
                )

        # 2. LocalStore status via PathRunLocator
        locator = PathRunLocator()
        location = locator.locate(run_root)

        # 3. If local final result exists and not override: publish immediately
        if location.is_final and not override:
            if progress_reporter is not None:
                progress_reporter.write(
                    f"Publishing existing local result for slot={slot_id}"
                )
            run_id = self.publish_to_mlflow(client, experiment_name, config, run_root)
            result = self._load_local_result(run_root)
            return W2VRunReceipt(
                result=result,
                run_id=run_id,
                staging_root=run_root,
                skipped=False,
            )

        # 4. If local checkpoint exists and not override: resume locally
        resume_from = None
        if location.is_checkpoint and not override:
            resume_from = location.checkpoint_path
            if progress_reporter is not None:
                progress_reporter.write(
                    f"Resuming {self.suite_name} slot={slot_id} from epoch {location.epoch} checkpoint"
                )

        # 5. Run training
        self.prepare_benchmarks(run_root, context.paths)
        result = self._run_training(
            config,
            context,
            run_root=run_root,
            local_store=local_store,
            resume_from=resume_from,
            progress_reporter=progress_reporter,
        )

        # 6. Publish to MLflow
        run_id = self.publish_to_mlflow(client, experiment_name, config, run_root)
        return W2VRunReceipt(
            result=result,
            run_id=run_id,
            staging_root=run_root,
            skipped=False,
        )

    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> Word2VecResult:
        identity = _mapping(config, "identity")
        run_key = str(identity["planned_run_slot_id"])
        root_override = context.metadata.get("run_root")
        root = (
            Path(root_override)
            if root_override
            else context.paths.runs("f2", self.suite_name, run_key)
        )
        if (
            root_override
            and Path(root_override).exists()
            and any(Path(root_override).iterdir())
        ):
            raise ValueError(f"staging directory is not empty: {root_override}")
        if not root_override and root.exists() and any(root.iterdir()):
            raise ValueError(f"staging directory is not empty: {root}")
        local_store = LocalStore(root)
        local_store.ensure_directories()
        self.prepare_benchmarks(root, context.paths)
        return self._run_training(
            config,
            context,
            run_root=root,
            local_store=local_store,
            resume_from=None,
            progress_reporter=context.metadata.get("progress_reporter"),
        )

    def prepare_benchmarks(self, run_root: Path, paths: Any) -> None:
        """Preserve available benchmark inputs for cache-only evaluation."""
        from f2.common.paths import get_benchmark_data_dir

        benchmarks = get_benchmark_data_dir(paths)
        for name in (
            "questions-words.txt",
            "questions-phrases.txt",
            "MSR-syntax.txt",
            "msr_sentence_completion/Holmes.lm_format.questions.txt",
            "msr_sentence_completion/Holmes.lm_format.answers.txt",
        ):
            source = benchmarks / name
            if source.is_file():
                target = run_root / "evaluation" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

    def _run_training(
        self,
        config: dict[str, object],
        context: ExperimentContext,
        *,
        run_root: Path,
        local_store: LocalStore,
        resume_from: Path | None = None,
        progress_reporter: Any | None = None,
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

        checkpoint_dir = local_store.checkpoint_dir
        prepared_dir = context.paths.inputs("f2", "prepared")

        session = create_session(
            config,
            context.paths.repo_root,
            cache_root=prepared_dir,
            corpus_digest=corpus_digest,
            resume_from=resume_from,
        )

        checkpoint_factory = (
            create_nnlm_checkpoint_manager
            if _mapping(config, "training")["model_kind"] == "feedforward_nnlm"
            else create_checkpoint_manager
        )
        manager = checkpoint_factory(
            checkpoint_dir,
            session=session,
            **(
                {
                    "vocabulary_artifact": vocabulary_cache_path(
                        corpus_digest,
                        _mapping(config, "vocabulary"),
                        cache_root=prepared_dir,
                        full_tokens="phrase_detection" in config,
                    ),
                }
                if checkpoint_factory is create_checkpoint_manager
                else {}
            ),
        )
        if resume_from is None:
            manager.save_latest()

        metrics_path = run_root / "metrics" / "observations.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not metrics_path.is_file():
            metrics_path.write_text(
                "epoch,processed_tokens,learning_rate,objective_loss_sum,objective_loss_count,objective_loss,elapsed_seconds,tokens_per_second\n",
                encoding="utf-8",
            )
        writer = DenseObservationWriter(metrics_path)
        total_epochs = int(_mapping(config, "training")["epochs"])

        existing_reports = []
        reports_file = run_root / "metrics" / "epoch_reports.json"
        if resume_from is not None and reports_file.is_file():
            try:
                existing_reports = json.loads(reports_file.read_text(encoding="utf-8"))
            except Exception:
                existing_reports = []

        if progress_reporter is not None:
            progress_reporter.set_total_updates(
                total_epochs, completed=session.completed_epochs
            )
            progress_reporter.write(
                f"preparing {self.suite_name} slot={identity.get('planned_run_slot_id', '')} "
                f"epochs={total_epochs} completed={session.completed_epochs}"
            )

        def on_observation(obs: list[dict[str, Any]]) -> None:
            writer.append(obs)

        def on_epoch_complete(
            epoch_num: int, epoch: Any, current_reports: list[dict[str, Any]]
        ) -> None:
            local_store.write_epoch_reports(current_reports)
            manager.save_latest()
            if progress_reporter is not None:
                progress_reporter.advance_to(epoch_num, epoch=epoch_num)

        engine = Word2VecEngine()
        exec_res = engine.train(
            session,
            total_epochs=total_epochs,
            on_epoch_complete=on_epoch_complete,
            on_observation=on_observation,
            progress_callback=progress_reporter.write if progress_reporter else None,
            existing_reports=existing_reports,
        )

        final = manager.save_final()
        if config.get("large_scale"):
            lookup_path = final.path
        else:
            lookup_path = run_root / "lookup"
            if lookup_path.exists():
                shutil.rmtree(lookup_path)
            if isinstance(session, TrainingSession):
                lookup_from_checkpoint(final.path, lookup_path)
            else:
                save_lookup_artifact(session.export_state(), lookup_path)

        report = {
            "identity": identity,
            "epochs": exec_res.reports,
            "epoch_history_complete": [item["epoch"] for item in exec_res.reports]
            == list(range(1, session.completed_epochs + 1)),
            "complete": session.is_complete,
            "artifacts": {
                "checkpoint": str(final.path.relative_to(run_root)),
                "lookup": str(lookup_path.relative_to(run_root)),
                "metrics": "metrics/observations.csv",
            },
        }
        report_path = local_store.write_result_report(report)
        local_store.write_config(config)
        return Word2VecResult(
            run_root, final.path, lookup_path, writer.path, report_path
        )

    def publish_to_mlflow(
        self,
        client: MlflowClient,
        experiment_name: str,
        config: dict[str, Any],
        run_root: Path,
    ) -> str:
        include_checkpoints = (
            bool(config.get("large_scale"))
            or _mapping(config, "training").get("model_kind") == "feedforward_nnlm"
        )
        return publish_run_to_mlflow(
            client,
            experiment_name,
            config,
            run_root,
            include_checkpoints=include_checkpoints,
        )

    def _load_local_result(self, run_root: Path) -> Word2VecResult:
        report_path = run_root / "result.json"
        data = json.loads(report_path.read_text(encoding="utf-8"))
        artifacts = data.get("artifacts", {})
        checkpoint = run_root / artifacts.get(
            "checkpoint", "checkpoints/generations/final"
        )
        lookup = run_root / artifacts.get("lookup", "lookup")
        metrics = run_root / artifacts.get("metrics", "metrics/observations.csv")
        return Word2VecResult(run_root, checkpoint, lookup, metrics, report_path)


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
    corpus_path = Path(str(_mapping(config, "corpus")["path"]))
    corpus = Corpus(
        corpus_path if corpus_path.is_absolute() else repo_root / corpus_path,
        full_tokens="phrase_detection" in config,
    )
    corpus_digest = corpus_digest or _corpus_digest(
        _mapping(config, "corpus"), Path(corpus.path)
    )
    vocabulary = resolve_or_build_vocabulary(
        corpus,
        _mapping(config, "vocabulary"),
        cache_root=cache_root or repo_root / "workspace/inputs/f2/prepared",
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

    if resume_from is not None:
        state = load_checkpoint(resume_from)
        model.restore_input_embeddings(state.input_embeddings())
        model.restore_output_embeddings(state.output_embeddings())
        return TrainingSession.restore(
            corpus, vocabulary, model, training, state, corpus_digest=corpus_digest
        )

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
    "W2VRunReceipt",
    "Word2VecExecutor",
    "Word2VecResult",
    "_corpus_digest",
    "_mapping",
    "create_session",
]
