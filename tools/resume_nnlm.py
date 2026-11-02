#!/usr/bin/env python3
"""Resume an interrupted NNLM Downpour training run from its epoch checkpoint.

Usage:
    # 1. Dry-run verification (loads checkpoint, restores session, checks validity without training)
    uv run python tools/resume_nnlm.py --dry-run

    # 2. Production execution under resource bound (e.g., optimal 10 physical cores):
    bound --cpu 0-4,10-14 --numa all --mem 64G --threads 10 run \
        uv run python tools/resume_nnlm.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from w2v import (
    Corpus,
    DownpourConfig,
    NnlmDownpourTrainingSession,
    NnlmTrainingConfig,
)

from f2.suites.w2v.artifacts import (
    resolve_or_build_vocabulary,
    save_lookup_artifact,
)
from f2.suites.w2v.executor import _mapping
from f2.suites.w2v.nnlm_artifacts import (
    create_checkpoint_manager,
    load_checkpoint,
)
from f2.suites.w2v.observations import DenseObservationWriter
from f2.suites.w2v.tracked import (
    _materialize_corpus,
    _publish,
    _verify_uploaded_manifest,
)
from f2.suites.w2v1.spec import parse_run_spec
from repro_core.context import RuntimePaths

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("resume_nnlm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume interrupted NNLM Downpour training from checkpoint."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="e39307353a574aa9a2ab20eb540d432e",
        help="MLflow run ID / staging directory name to resume.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="studies/f2/src/f2/suites/w2v1/config/e02_table3.yaml",
        help="Path to the experiment specification YAML.",
    )
    parser.add_argument(
        "--atomic-run-id",
        type=str,
        default="wmt--nnlm-d640-h640-n8-w320m",
        help="Atomic variant identifier within the spec.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for execution identity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load checkpoint and verify restoration without executing training.",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Skip publishing artifacts to MLflow upon completion.",
    )
    return parser.parse_args()


def resume_training(
    run_id: str,
    config_path: Path,
    atomic_run_id: str,
    seed: int,
    *,
    dry_run: bool = False,
    publish: bool = True,
) -> None:
    paths = RuntimePaths.from_environment()
    run_root = paths.staging_root / "exp" / "f2" / "w2v1" / "tracked" / run_id

    if not run_root.is_dir():
        raise FileNotFoundError(
            f"Staging directory for run {run_id} not found: {run_root}"
        )

    checkpoints_dir = run_root / "checkpoints"
    latest_meta = checkpoints_dir / "latest.json"
    if not latest_meta.is_file():
        raise FileNotFoundError(f"Checkpoint metadata not found: {latest_meta}")

    latest_info = json.loads(latest_meta.read_text())
    ckpt_rel_path = latest_info.get("path")
    if not ckpt_rel_path:
        raise ValueError(f"Invalid latest.json: missing 'path' key: {latest_info}")

    ckpt_dir = checkpoints_dir / ckpt_rel_path
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    logger.info(
        "Found latest checkpoint: %s (epoch %d, update %d)",
        ckpt_dir.name,
        latest_info.get("epoch", -1),
        latest_info.get("update", -1),
    )

    # 1. Load checkpoint state
    logger.info("Loading NNLM checkpoint state...")
    t0 = time.perf_counter()
    state = load_checkpoint(ckpt_dir)
    logger.info(
        "Loaded state in %.2fs. Completed epochs: %d, tokens: %d",
        time.perf_counter() - t0,
        state.completed_epochs,
        state.processed_tokens,
    )

    # 2. Reconstruct config and corpus bindings
    abs_config_path = (
        config_path if config_path.is_absolute() else paths.repo_root / config_path
    )
    spec = parse_run_spec(abs_config_path, atomic_run_id=atomic_run_id).with_seed(seed)
    config = spec.to_executor_config()

    logger.info("Materializing verified corpus...")
    _materialize_corpus(config, paths)
    corpus_info = _mapping(config, "corpus")
    corpus_path = Path(str(corpus_info["path"]))
    corpus_sha256 = str(corpus_info["sha256"])

    corpus = Corpus(corpus_path)
    vocabulary = resolve_or_build_vocabulary(
        corpus,
        _mapping(config, "vocabulary"),
        cache_root=paths.cache_root,
        corpus_digest=corpus_sha256,
    )

    training_vals = dict(_mapping(config, "training"))
    training_vals["root_seed"] = seed
    training_vals.pop("model_kind", None)
    training = NnlmTrainingConfig(**training_vals)

    dist_vals = dict(_mapping(config, "distribution"))
    dist_vals.pop("mode", None)
    downpour = DownpourConfig(**dist_vals)

    # 3. Restore session
    logger.info("Restoring NnlmDownpourTrainingSession...")
    session = NnlmDownpourTrainingSession.restore(
        corpus, vocabulary, training, downpour, state, corpus_digest=corpus_sha256
    )
    logger.info(
        "Successfully restored training session! completed_epochs=%d, processed_tokens=%d, is_complete=%s",
        session.completed_epochs,
        session.processed_tokens,
        session.is_complete,
    )

    if dry_run:
        logger.info(
            "Dry-run mode requested. Checkpoint validation and session restoration succeeded."
        )
        return

    if session.is_complete:
        logger.info(
            "Session is already marked complete (completed_epochs >= total_epochs)."
        )
        return

    total_epochs = int(_mapping(config, "training")["epochs"])
    logger.info(
        "Resuming training from epoch %d to %d...",
        session.completed_epochs + 1,
        total_epochs,
    )

    # 4. Attach checkpoint manager and observations writer
    manager = create_checkpoint_manager(checkpoints_dir, session=session)
    obs_file = run_root / "metrics" / "observations.csv"
    writer = DenseObservationWriter(obs_file)

    reports = []
    # 4.1 Reconstruct prior completed epoch summaries from existing observations.csv
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
            if ep <= session.completed_epochs:
                reports.append(prior_by_epoch[ep])
                logger.info(
                    "Loaded previous summary for Epoch %d: tokens=%d, loss=%.4f",
                    ep,
                    prior_by_epoch[ep]["processed_tokens"],
                    prior_by_epoch[ep]["objective_loss"],
                )

    while not session.is_complete:
        current_target_epoch = session.completed_epochs + 1
        logger.info(">>> Starting Epoch %d/%d...", current_target_epoch, total_epochs)
        ep_t0 = time.perf_counter()
        epoch = session.train_epoch()
        ep_duration = time.perf_counter() - ep_t0

        writer.append(epoch.observations())
        manager.save_latest()

        logger.info(
            "=== Finished Epoch %d/%d: tokens=%d, loss=%.4f, throughput=%.1f tokens/s (wall: %.1fs)",
            epoch.epoch,
            total_epochs,
            epoch.processed_tokens,
            epoch.objective_loss,
            epoch.tokens_per_second,
            ep_duration,
        )
        reports.append(
            {
                "epoch": epoch.epoch,
                "processed_tokens": epoch.processed_tokens,
                "objective_loss": epoch.objective_loss,
            }
        )

    # 5. Finalize artifacts
    logger.info("All epochs completed. Saving final checkpoint and lookup artifact...")
    final = manager.save_final()
    final_state = session.export_state()
    lookup_path = run_root / "lookup"
    save_lookup_artifact(final_state, lookup_path)

    report = {
        "identity": _mapping(config, "identity"),
        "epochs": reports,
        "complete": session.is_complete,
        "artifacts": {
            "checkpoint": str(final.path.relative_to(run_root)),
            "lookup": str(lookup_path.relative_to(run_root)),
            "metrics": "metrics/observations.csv",
        },
    }
    report_path = run_root / "result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info("Saved result summary to %s", report_path)

    # 6. Publish to MLflow
    if publish:
        tracking_uri = os.environ.get(
            "F2_MLFLOW_TRACKING_URI", "http://127.0.0.1:15002/mlflow-f2/"
        )
        logger.info("Publishing completed artifacts to MLflow at %s...", tracking_uri)
        try:
            from mlflow import MlflowClient

            client = MlflowClient(tracking_uri=tracking_uri)
            run_info = client.get_run(run_id).info
            if run_info.lifecycle_stage == "deleted":
                logger.info("Run was in deleted state; restoring to active...")
                client.restore_run(run_id)

            _publish(client, run_id, run_root, include_checkpoints=True)
            _verify_uploaded_manifest(client, run_id)
            client.set_tag(run_id, "trial.status", "finished")
            client.set_tag(run_id, "result.durable_complete", "true")
            client.set_terminated(run_id, status="FINISHED")
            logger.info(
                "Successfully published and closed MLflow run %s (FINISHED).", run_id
            )
        except Exception as e:
            logger.error(
                "Failed to publish to MLflow: %s. Local artifacts are preserved.", e
            )
            raise


def main() -> None:
    args = parse_args()
    resume_training(
        run_id=args.run_id,
        config_path=Path(args.config_path),
        atomic_run_id=args.atomic_run_id,
        seed=args.seed,
        dry_run=args.dry_run,
        publish=not args.no_publish,
    )


if __name__ == "__main__":
    main()
