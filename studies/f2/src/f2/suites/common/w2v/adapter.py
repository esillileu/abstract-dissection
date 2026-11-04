"""Word2Vec training session execution and adapter routines."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from f2.common.paths import get_benchmark_data_dir
from f2.execution.coordinator import (
    ExecutionDecisionKind,
    F2ExecutionCoordinator,
)
from f2.execution.identity import ConditionIdentity
from f2.execution.resources import resolve_and_prepare_corpus
from f2.suites.common.w2v.artifacts import (
    create_checkpoint_manager,
    lookup_from_checkpoint,
    save_lookup_artifact,
    vocabulary_cache_path,
)
from f2.suites.common.w2v.engine import Word2VecEngine
from f2.suites.common.w2v.nnlm_artifacts import (
    create_checkpoint_manager as create_nnlm_checkpoint_manager,
)
from f2.suites.common.w2v.observations import DenseObservationWriter
from f2.suites.common.w2v.training import (
    W2VRunReceipt,
    Word2VecResult,
    corpus_digest,
    create_session,
    mapping,
)
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.storage.local_store import LocalStore


def prepare_benchmarks(run_root: Path, paths: RuntimePaths) -> None:
    """Preserve available benchmark inputs for cache-only evaluation."""
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


def execute_w2v_training(
    config: dict[str, object],
    context: ExperimentContext,
    *,
    suite_name: str,
    run_root: Path,
    local_store: LocalStore,
    resume_from: Path | None = None,
    progress_reporter: Any | None = None,
) -> Word2VecResult:
    """Execute raw Word2Vec numerical training for the given session config."""
    prepare_benchmarks(run_root, context.paths)
    identity = mapping(config, "identity")
    corpus_config = mapping(config, "corpus")
    source = Path(str(corpus_config["path"]))
    if not source.is_absolute():
        source = context.paths.repo_root / source
    if not source.is_file():
        raise ValueError(f"{suite_name.upper()} corpus does not exist: {source}")
    c_digest = corpus_digest(corpus_config, source)

    checkpoint_dir = local_store.checkpoint_dir
    prepared_dir = context.paths.inputs("f2", "prepared")

    session = create_session(
        config,
        context.paths.repo_root,
        cache_root=prepared_dir,
        corpus_digest=c_digest,
        resume_from=resume_from,
    )

    checkpoint_factory = (
        create_nnlm_checkpoint_manager
        if mapping(config, "training")["model_kind"] == "feedforward_nnlm"
        else create_checkpoint_manager
    )
    manager = checkpoint_factory(
        checkpoint_dir,
        session=session,
        **(
            {
                "vocabulary_artifact": vocabulary_cache_path(
                    c_digest,
                    mapping(config, "vocabulary"),
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
    total_epochs = int(mapping(config, "training")["epochs"])

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
            f"preparing {suite_name} slot={identity.get('planned_run_slot_id', '')} "
            f"epochs={total_epochs} completed={session.completed_epochs}"
        )

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
        on_observation=writer.append,
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
        from w2v import TrainingSession

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
    return Word2VecResult(run_root, final.path, lookup_path, writer.path, report_path)


def load_local_w2v_result(run_root: Path) -> Word2VecResult:
    report_path = run_root / "result.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts = data.get("artifacts", {})
    checkpoint = run_root / artifacts.get("checkpoint", "checkpoints/generations/final")
    lookup = run_root / artifacts.get("lookup", "lookup")
    metrics = run_root / artifacts.get("metrics", "metrics/observations.csv")
    return Word2VecResult(run_root, checkpoint, lookup, metrics, report_path)


def run_w2v_with_coordinator(
    config: dict[str, object],
    *,
    suite_name: str,
    tracking_uri: str,
    paths: RuntimePaths,
    override: bool = False,
    progress_reporter: Any | None = None,
    training_step: Any | None = None,
) -> W2VRunReceipt:
    """Coordinate the F2 run lifecycle (skip/publish/resume/run/publish) via F2ExecutionCoordinator."""
    coordinator = F2ExecutionCoordinator(tracking_uri=tracking_uri, paths=paths)
    condition = ConditionIdentity.from_config(config, suite_id=suite_name)
    experiment_name = str(
        mapping(config, "tracking").get("experiment", f"f2.{suite_name}")
    )

    decision = coordinator.decide(
        condition,
        experiment_name=experiment_name,
        override=override,
    )

    if decision.kind in (ExecutionDecisionKind.RUN, ExecutionDecisionKind.RESUME):
        resolve_and_prepare_corpus(config, paths, progress_reporter=progress_reporter)

    context = ExperimentContext(
        paths=paths,
        metadata={"progress_reporter": progress_reporter},
    )

    step_fn = training_step or (
        lambda cfg, ctx, *, run_root, local_store, resume_from=None, progress_reporter=None: (
            execute_w2v_training(
                cfg,
                ctx,
                suite_name=suite_name,
                run_root=run_root,
                local_store=local_store,
                resume_from=resume_from,
                progress_reporter=progress_reporter,
            )
        )
    )

    receipt = coordinator.execute_condition(
        condition,
        config,
        context,
        experiment_name=experiment_name,
        training_delegate=step_fn,
        override=override,
        progress_reporter=progress_reporter,
        load_result_delegate=load_local_w2v_result,
        decision=decision,
    )
    return W2VRunReceipt(
        result=receipt.result,
        run_id=receipt.run_id,
        staging_root=receipt.run_dir,
        skipped=receipt.skipped,
    )


__all__ = [
    "execute_w2v_training",
    "load_local_w2v_result",
    "prepare_benchmarks",
    "run_w2v_with_coordinator",
]
