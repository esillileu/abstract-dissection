from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from deepscratch.core import Tensor, configure_runtime, seed_batch_order
from deepscratch.profiling import create_runtime_monitor, training_summary
from deepscratch.trainer import LanguageModelTrainer

from dlfs.adapters.checkpoint import create_deepscratch_checkpoint_manager
from dlfs.ds2.implemented import executor
from dlfs.ds2.implemented.adapters import (
    build_language_model,
    build_sequence_objective,
    build_sequence_optimizer,
    language_model_training_corpus,
)
from repro_core.context import ExperimentContext
from repro_core.context.checkpoint import CheckpointRetentionPolicy
from repro_core.context.contracts import ExperimentResult
from repro_core.context.event_executor import EvaluationRequest, EventExperimentExecutor

from ..records import DS2Records
from .checkpoint import _config_digest, _record_retained_checkpoints, _save_epoch_roles
from .common import (
    _apply_validation_decay,
    _artifact_root,
    _backend_exp_float,
    _device_timer,
    _final,
    _mapping,
    _source_curve_from_objective,
)
from .digest import _array_digest


class LanguageModelExecutor:
    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        ptb = executor.load_ds2_ptb()
        model_config, loader, training = (
            _mapping(config, key) for key in ("model", "loader", "training")
        )
        dataset, evaluation = (
            _mapping(config, "dataset"),
            _mapping(config, "evaluation"),
        )
        train_corpus, vocab_size = language_model_training_corpus(ptb["train"], dataset)
        model = build_language_model(
            str(model_config.get("name")),
            vocab_size,
            model_config,
            backend,
            dropout_rng=backend.random_stream("dropout"),
        )
        objective = build_sequence_objective(_mapping(config, "objective"), backend)
        optimizer = build_sequence_optimizer(config, model, objective)
        seed_batch_order(backend, streams)
        train = Tensor(
            backend.xp.asarray(train_corpus[:-1], dtype=backend.xp.int64),
            backend=backend,
        )
        train_targets = Tensor(
            backend.xp.asarray(train_corpus[1:], dtype=backend.xp.int64),
            backend=backend,
        )
        max_epochs = int(training.get("max_epochs", 4))
        max_updates = training.get("max_updates")
        valid = Tensor(
            backend.xp.asarray(ptb["valid"][:-1], dtype=backend.xp.int64),
            backend=backend,
        )
        valid_targets = Tensor(
            backend.xp.asarray(ptb["valid"][1:], dtype=backend.xp.int64),
            backend=backend,
        )
        test = Tensor(
            backend.xp.asarray(ptb["test"][:-1], dtype=backend.xp.int64),
            backend=backend,
        )
        test_targets = Tensor(
            backend.xp.asarray(ptb["test"][1:], dtype=backend.xp.int64), backend=backend
        )
        context.metadata["data"] = {
            "dataset_checksum": _array_digest(ptb["train"], ptb["valid"], ptb["test"]),
            "split_checksum": _array_digest(train_corpus, ptb["valid"], ptb["test"]),
        }
        valid_ppl = float("inf")
        test_ppl = float("inf")
        best_valid = float("inf")
        best_valid_epoch = 0
        valid_every_epochs = int(evaluation.get("valid_every_epochs", 1))
        test_every_epochs = int(evaluation.get("test_every_epochs", 1))
        test_at_end = bool(evaluation.get("test_at_end", False))
        requests = {
            "valid": EvaluationRequest(
                "ptb-valid", "valid", (valid, valid_targets), ("perplexity",)
            ),
            "test": EvaluationRequest(
                "ptb-test", "test", (test, test_targets), ("perplexity",)
            ),
        }
        trainer = LanguageModelTrainer(
            model,
            objective,
            optimizer,
            max_epochs=max_epochs,
            batch_size=int(loader.get("batch_size", 20)),
            time_size=int(loader.get("time_size", 35)),
            max_updates=None if max_updates is None else int(max_updates),
            epoch_cursor=str(
                _mapping(config, "policy").get("epoch_cursor", "continuous")
            ),
            epoch_recurrent_state=str(
                _mapping(config, "policy").get("epoch_recurrent_state", "continuous")
            ),
            evaluator_batch_size=int(
                _mapping(config, "evaluation").get("batch_size", 10)
            ),
            evaluator_time_size=int(
                _mapping(config, "evaluation").get("time_size", 35)
            ),
            evaluator_drop_remainder=bool(
                _mapping(config, "evaluation").get("drop_remainder", True)
            ),
        )
        checkpoint_manager = create_deepscratch_checkpoint_manager(
            Path(str(context.metadata["checkpoint_root"])),
            model=model,
            objective=objective,
            optimizer=optimizer,
            trainer=trainer,
            config_digest=_config_digest(config),
            policy=CheckpointRetentionPolicy.from_mapping(
                _mapping(config, "checkpoint")
            ),
        )

        def evaluate_request(request):
            return trainer.evaluate(*request.source)

        def epoch_requests(event):
            values = []
            if valid_every_epochs > 0 and event.epoch % valid_every_epochs == 0:
                values.append(requests["valid"])
            if test_every_epochs > 0 and event.epoch % test_every_epochs == 0:
                values.append(requests["test"])
            return tuple(values)

        artifact_root = _artifact_root(config, context)
        records_sink = DS2Records()
        records_sink.bind_artifact_root(artifact_root)
        monitor = create_runtime_monitor(backend, _mapping(config, "profiling"))

        def after_evaluation(request, result, _axis, step):
            nonlocal valid_ppl, test_ppl, best_valid, best_valid_epoch
            if request.split == "valid":
                valid_ppl = float(result.perplexity)
                if valid_ppl < best_valid:
                    best_valid, best_valid_epoch = valid_ppl, step
                    if bool(_mapping(config, "checkpoint").get("save_best", False)):
                        records_sink.flush()
                        checkpoint_manager.save_best()
                else:
                    _apply_validation_decay(config, optimizer)
            else:
                test_ppl = float(result.perplexity)

        controller = EventExperimentExecutor(
            records=records_sink,
            evaluate=evaluate_request,
            epoch_requests=epoch_requests,
            after_evaluation=after_evaluation,
            source_curve=_source_curve_from_objective(config),
            after_epoch=lambda _event: _save_epoch_roles(checkpoint_manager),
            device_timer=_device_timer(config, backend),
            progress=context.metadata.get("progress_reporter"),
        )
        trainer.event_receivers = (controller,)
        progress = context.metadata.get("progress_reporter")
        if progress is not None:
            progress.set_total_updates(
                trainer.planned_total_updates(len(train)),
                completed=trainer.global_step,
            )
        with training_summary(monitor):
            records = controller.run(lambda: trainer.fit(train, train_targets))
        if test_at_end:
            test_ppl = float(trainer.evaluate(test, test_targets).perplexity)
        checkpoint_manager.save_final()
        _record_retained_checkpoints(
            records,
            checkpoint_manager,
            best_metric="valid/perplexity",
            best_value=None if best_valid == float("inf") else best_valid,
        )
        records.flush()
        final_train_ppl = (
            _backend_exp_float(backend, records.updates[-1]["loss"])
            if records.updates
            else float("inf")
        )
        final_metrics = {
            "final/train/perplexity": final_train_ppl,
            "final/train/ppl": final_train_ppl,
        }
        if test_ppl < float("inf"):
            final_metrics.update(
                {
                    "final/test/perplexity": test_ppl,
                    "final/test/ppl": test_ppl,
                }
            )
        if best_valid < float("inf"):
            final_metrics.update(
                {
                    "final/valid/perplexity": valid_ppl,
                    "final/valid/ppl": valid_ppl,
                    "final/best_valid_ppl": best_valid,
                    "final/best_valid_epoch": float(best_valid_epoch),
                }
            )
        profiling_metrics = monitor.metrics()
        return ExperimentResult(
            metrics=_final(
                updates=trainer.global_step,
                epochs=trainer.epoch,
                samples=len(train) * trainer.epoch,
                **final_metrics,
            ),
            artifact_root=artifact_root,
            model=model,
            metric_rows=records.mlflow_metric_rows(),
            profiling_metrics=profiling_metrics,
        )
