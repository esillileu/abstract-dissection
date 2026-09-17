from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from deepscratch.core import Tensor, configure_runtime, seed_batch_order
from deepscratch.profiling import create_runtime_monitor, training_summary
from deepscratch.trainer import Seq2seqTrainer

from dlfs.adapters.checkpoint import create_deepscratch_checkpoint_manager
from dlfs.ds2.implemented import executor
from dlfs.ds2.implemented.adapters import (
    build_seq2seq_model,
    build_sequence_objective,
    build_sequence_optimizer,
)
from repro_core.context import ExperimentContext
from repro_core.context.checkpoint import CheckpointRetentionPolicy
from repro_core.context.contracts import ExperimentResult
from repro_core.context.event_executor import EvaluationRequest, EventExperimentExecutor

from ..records import DS2Records
from .checkpoint import _config_digest, _record_retained_checkpoints, _save_epoch_roles
from .common import (
    _artifact_root,
    _device_timer,
    _final,
    _mapping,
    _recorded_float,
)
from .digest import _array_digest, _file_digest, _sequence_dataset_path
from .seq_predictions import _record_seq_predictions


class Seq2SeqExecutor:
    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset, model_config, loader, training = (
            _mapping(config, key) for key in ("dataset", "model", "loader", "training")
        )
        split_seed = int(dataset.get("split_seed", streams.dataset_split))
        split_algorithm = str(dataset.get("split_algorithm", "default_rng"))
        data = executor.load_ds2_sequence(
            str(dataset["file"]),
            seed=split_seed,
            split_algorithm=split_algorithm,
        )
        x_train, t_train = data["train"]
        x_test, t_test = data["test"]
        if bool(dataset.get("reverse", False)):
            x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
        context.metadata["data"] = {
            "split_seed": split_seed,
            "split_algorithm": split_algorithm,
            "dataset_checksum": _file_digest(
                _sequence_dataset_path(str(dataset["file"]))
            ),
            "split_checksum": _array_digest(x_train, t_train, x_test, t_test),
        }
        model = build_seq2seq_model(
            str(model_config.get("name")),
            len(data["char_to_id"]),
            model_config,
            backend,
        )
        objective = build_sequence_objective(_mapping(config, "objective"), backend)
        optimizer = build_sequence_optimizer(config, model, objective)
        seed_batch_order(backend, streams)
        batch_size, epochs = (
            int(loader.get("batch_size", 128)),
            int(training.get("max_epochs", 10)),
        )
        eval_batch_size = int(loader.get("eval_batch_size", batch_size))
        max_updates = training.get("max_updates")
        train_x = Tensor(
            backend.xp.asarray(x_train, dtype=backend.xp.int64), backend=backend
        )
        train_t = Tensor(
            backend.xp.asarray(t_train, dtype=backend.xp.int64), backend=backend
        )
        test_source = (
            Tensor(backend.xp.asarray(x_test, dtype=backend.xp.int64), backend=backend),
            Tensor(backend.xp.asarray(t_test, dtype=backend.xp.int64), backend=backend),
        )
        request = EvaluationRequest(
            "sequence-test-full",
            "test",
            test_source,
            ("exact_match_accuracy", "token_accuracy"),
        )
        evaluation_config = _mapping(config, "evaluation")
        test_every_epochs = int(evaluation_config.get("test_every_epochs", 1))
        trainer = Seq2seqTrainer(
            model,
            objective,
            optimizer,
            max_epochs=epochs,
            batch_size=batch_size,
            start_id=data["char_to_id"]["_"],
            eval_batch_size=eval_batch_size,
            max_updates=None if max_updates is None else int(max_updates),
            drop_last=bool(loader.get("drop_last", False)),
            loss_timing=str(
                _mapping(config, "policy").get("loss_timing", "post_update")
            ),
            batch_rng=backend.random_stream("batch_order"),
        )
        artifact_root = _artifact_root(config, context)
        records = DS2Records()
        records.bind_artifact_root(artifact_root)
        monitor = create_runtime_monitor(backend, _mapping(config, "profiling"))
        checkpoint_config = _mapping(config, "checkpoint")
        checkpoint_manager = create_deepscratch_checkpoint_manager(
            Path(str(context.metadata["checkpoint_root"])),
            model=model,
            objective=objective,
            optimizer=optimizer,
            trainer=trainer,
            config_digest=_config_digest(config),
            policy=CheckpointRetentionPolicy.from_mapping(checkpoint_config),
        )
        best_exact = -1.0

        def after_evaluation(_request, result, axis, step):
            nonlocal best_exact
            if axis == "epoch" and result.exact_match_accuracy is not None:
                records.add_source_curve(
                    {
                        "series_id": "full_test_exact_match",
                        "plot_index": step - 1,
                        "update_start": trainer.global_step,
                        "update_end": trainer.global_step,
                        "epoch_start": step,
                        "epoch_end": step,
                        "unit": "sequence",
                        "unit_count": result.example_count,
                        "metric": "exact_match_accuracy",
                        "reducer": "identity",
                        "value": result.exact_match_accuracy,
                    }
                )
                _record_seq_predictions(
                    records,
                    model,
                    x_test,
                    t_test,
                    data["char_to_id"],
                    data["id_to_char"],
                    backend,
                    _mapping(config, "recording"),
                    epoch=step,
                    predictions=trainer.last_predictions,
                )
                if (
                    bool(checkpoint_config.get("save_best", False))
                    and result.exact_match_accuracy > best_exact
                ):
                    best_exact = float(result.exact_match_accuracy)
                    records.flush()
                    checkpoint_manager.save_best()

        controller = EventExperimentExecutor(
            records=records,
            evaluate=lambda _request: trainer.evaluate(
                *test_source, metrics=("exact_match_accuracy", "token_accuracy")
            ),
            epoch_requests=lambda event: (
                (request,)
                if test_every_epochs > 0 and event.epoch % test_every_epochs == 0
                else ()
            ),
            after_evaluation=after_evaluation,
            after_epoch=lambda _event: _save_epoch_roles(checkpoint_manager),
            device_timer=_device_timer(config, backend),
            progress=context.metadata.get("progress_reporter"),
        )
        trainer.event_receivers = (controller,)
        progress = context.metadata.get("progress_reporter")
        if progress is not None:
            progress.set_total_updates(
                trainer.planned_total_updates(len(train_x)),
                completed=trainer.global_step,
            )
        with training_summary(monitor):
            records = controller.run(lambda: trainer.fit(train_x, train_t))
        _record_retained_checkpoints(
            records,
            checkpoint_manager,
            best_metric="test/exact_match_accuracy",
            best_value=None if best_exact < 0 else best_exact,
        )
        selected = checkpoint_manager.current("best")
        record_checkpoint = context.metadata.get("record_eval_checkpoint")
        if selected is not None and callable(record_checkpoint):
            record_checkpoint(selected.path)
        records.flush()
        last_evaluation = (
            records.evaluations[-2:] if len(records.evaluations) >= 2 else []
        )
        final_values = {
            "final/train/loss": _recorded_float(records.updates[-1]["loss"])
            if records.updates
            else 0.0
        }
        for row in last_evaluation:
            final_values[f"final/test/{row['metric'].replace('_accuracy', '')}"] = (
                float(row["value"])
            )
        return ExperimentResult(
            metrics=_final(
                updates=trainer.global_step,
                epochs=trainer.epoch,
                samples=trainer.samples_seen,
                **final_values,
            ),
            artifact_root=artifact_root,
            model=model,
            metric_rows=records.mlflow_metric_rows(),
            profiling_metrics=monitor.metrics(),
        )
