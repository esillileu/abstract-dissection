from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from deepscratch.core import Tensor, configure_runtime, seed_batch_order
from deepscratch.profiling import create_runtime_monitor, training_summary
from deepscratch.trainer import FusedNegativeSamplingTrainer, Word2VecTrainer

from dlfs.adapters.checkpoint import create_deepscratch_checkpoint_manager
from dlfs.ds2.implemented.adapters import (
    build_sequence_optimizer,
    build_unigram_sampler,
    build_word2vec_batch_adapter,
    build_word2vec_model,
    build_word2vec_objective,
    contexts_targets,
    load_ds2_word2vec_corpus,
)
from repro_core.context import ExperimentContext
from repro_core.context.checkpoint import CheckpointRetentionPolicy
from repro_core.context.contracts import ExperimentResult
from repro_core.context.event_executor import EventExperimentExecutor

from ..records import DS2Records
from .checkpoint import _config_digest, _record_retained_checkpoints, _save_epoch_roles
from .common import (
    _artifact_root,
    _device_timer,
    _final,
    _mapping,
    _recorded_float,
    _source_curve_from_objective,
)
from .digest import _array_digest


class Word2VecExecutor:
    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset, model_config, objective_config, training = (
            _mapping(config, key)
            for key in ("dataset", "model", "objective", "training")
        )
        corpus, word_to_id = load_ds2_word2vec_corpus(_mapping(config, "dataset"))
        window = int(dataset.get("window_size", 5))
        contexts, targets = contexts_targets(corpus, window)
        context.metadata["data"] = {
            "dataset_checksum": _array_digest(corpus),
            "split_checksum": _array_digest(contexts, targets),
        }
        objective_name = str(objective_config.get("name", "NegativeSampling"))
        sampler = None
        if objective_name in {"NegativeSampling", "FusedNegativeSampling"}:
            sampler_values = _mapping(objective_config, "sampler")
            sampler = build_unigram_sampler(
                sampler_values,
                corpus,
                vocab_size=len(word_to_id),
                backend=backend,
                rng=backend.random_stream("negative_sampling"),
            )
            context.metadata["negative_sampler"] = sampler.metadata
        architecture = str(model_config.get("name", "CBOW"))
        input_representation = str(
            model_config.get("input_representation", "embedding")
        )
        embedding_size = int(model_config.get("embedding_size", 100))
        model = build_word2vec_model(
            architecture,
            input_representation,
            len(word_to_id),
            embedding_size,
            backend=backend,
        )
        adapter = build_word2vec_batch_adapter(
            architecture,
            input_representation,
            len(word_to_id),
            objective_name,
        )
        objective = build_word2vec_objective(
            objective_name,
            objective_config,
            len(word_to_id),
            sampler,
            backend=backend,
            is_skipgram=architecture == "SkipGram",
        )
        optimizer = build_sequence_optimizer(config, model, objective)
        seed_batch_order(backend, streams)
        loader = _mapping(config, "loader")
        batch_size, epochs = (
            int(loader.get("batch_size", 100)),
            int(training.get("max_epochs", 10)),
        )
        max_updates = training.get("max_updates")
        x = backend.xp.asarray(contexts, dtype=backend.xp.int64)
        t = backend.xp.asarray(targets, dtype=backend.xp.int64)
        artifact_root = _artifact_root(config, context)
        records_sink = DS2Records()
        records_sink.bind_artifact_root(artifact_root)
        monitor = create_runtime_monitor(backend, _mapping(config, "profiling"))
        trainer_type = (
            FusedNegativeSamplingTrainer
            if objective_name == "FusedNegativeSampling"
            else Word2VecTrainer
        )
        trainer = trainer_type(
            model,
            objective,
            optimizer,
            batch_adapter=adapter,
            max_epochs=epochs,
            batch_size=batch_size,
            max_updates=None if max_updates is None else int(max_updates),
            drop_last=bool(loader.get("drop_last", True)),
            event_receivers=[],
            batch_rng=backend.random_stream("batch_order"),
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
        controller = EventExperimentExecutor(
            records=records_sink,
            evaluate=lambda _request: None,
            source_curve=_source_curve_from_objective(config),
            after_epoch=lambda _event: _save_epoch_roles(checkpoint_manager),
            device_timer=_device_timer(config, backend),
            progress=context.metadata.get("progress_reporter"),
        )
        trainer.event_receivers = (controller,)
        progress = context.metadata.get("progress_reporter")
        if progress is not None:
            progress.set_total_updates(
                trainer.planned_total_updates(len(x)),
                completed=trainer.global_step,
            )
        with training_summary(
            monitor,
            synchronize=bool(
                _mapping(config, "profiling").get(
                    "synchronize_train",
                    False,
                )
            ),
        ):
            records = controller.run(
                lambda: trainer.fit(
                    Tensor(x, backend=backend), Tensor(t, backend=backend)
                ),
                start_update=trainer.global_step + 1,
            )
        _record_retained_checkpoints(records, checkpoint_manager)
        records.flush()
        final_loss = (
            _recorded_float(records.updates[-1]["loss"]) if records.updates else 0.0
        )
        final_metrics = {"final/train/loss": final_loss}
        if records.updates and records.updates[-1].get("book_loss") is not None:
            final_metrics["final/train/book_loss"] = _recorded_float(
                records.updates[-1]["book_loss"]
            )
        profiling_metrics = monitor.metrics()
        return ExperimentResult(
            metrics=_final(
                updates=trainer.global_step,
                epochs=trainer.epoch,
                samples=len(x) * trainer.epoch,
                **final_metrics,
            ),
            artifact_root=artifact_root,
            model=model,
            metric_rows=records.mlflow_metric_rows(),
            profiling_metrics=profiling_metrics,
        )
