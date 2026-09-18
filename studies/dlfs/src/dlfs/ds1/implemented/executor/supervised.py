from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from deepscratch.core import configure_runtime, seed_batch_order
from deepscratch.datasets.mnist import save_file as mnist_cache_path
from deepscratch.nn.layers import BatchNormalization
from deepscratch.profiling import create_runtime_monitor, training_summary
from deepscratch.trainer import ForwardTrainer

from dlfs.ds1.implemented.adapters import (
    build_ds1_model,
    build_ds1_objective,
    build_ds1_optimizer,
    training_parameters,
)
from dlfs.ds1.implemented.adapters import (
    load_ds1_mnist as _default_load_ds1_mnist,
)
from repro_core.context import ExperimentContext
from repro_core.context.checkpoint import CheckpointManager
from repro_core.context.contracts import ExperimentResult
from repro_core.context.metrics import build_final_metrics

from ..final_gap import evaluate_checkpoint_gap, is_target_run
from ..records import DS1Records
from .checkpointing import (
    compute_checkpoint_identity_digest,
    record_manager_checkpoints,
    setup_checkpoint_manager,
)
from .common import (
    _artifact_root,
    _mapping,
    _path_digest,
)
from .events import create_supervised_events
from .schedule import _evaluation_requests
from .transforms import (
    _apply_input_transform,
    _permute_pixels,
    _validation_probe,
)


def _get_load_ds1_mnist():
    mod = sys.modules.get("dlfs.ds1.implemented.executor")
    if mod is not None and hasattr(mod, "load_ds1_mnist"):
        return mod.load_ds1_mnist
    return _default_load_ds1_mnist


class SupervisedClassificationExecutor:
    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        (
            dataset,
            model_config,
            objective_config,
            training_config,
            loader_config,
            optimizer_config,
        ) = (
            _mapping(config, key)
            for key in (
                "dataset",
                "model",
                "objective",
                "training",
                "loader",
                "optimizer",
            )
        )
        backend, streams, actual_runtime = configure_runtime(config)
        context.metadata["runtime"] = actual_runtime
        context.metadata["seed_streams"] = asdict(streams)
        flatten = bool(dataset.get("flatten", True))
        gpu = backend.is_gpu
        load_mnist = _get_load_ds1_mnist()
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=flatten, gpu=gpu)
        if (limit := dataset.get("train_limit")) is not None:
            x_train, t_train = x_train[: int(limit)], t_train[: int(limit)]
        if (limit := dataset.get("test_limit")) is not None:
            x_test, t_test = x_test[: int(limit)], t_test[: int(limit)]
        transform_metadata = _apply_input_transform(
            dataset=dataset,
            backend=backend,
            x_train=x_train,
            x_test=x_test,
            artifact_root=Path(str(context.metadata["artifact_root"])),
        )
        if transform_metadata["name"] == "pixel_permutation":
            permutation = transform_metadata.pop("permutation")
            x_train, x_test = (
                _permute_pixels(x_train, permutation),
                _permute_pixels(x_test, permutation),
            )
        x_train, t_train, x_valid, t_valid, validation_metadata = _validation_probe(
            dataset=dataset,
            backend=backend,
            x_train=x_train,
            t_train=t_train,
            artifact_root=Path(str(context.metadata["artifact_root"])),
        )
        context.metadata["data"] = {
            "cache_path": mnist_cache_path,
            "cache_sha256": _path_digest(Path(mnist_cache_path)),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "flatten": flatten,
            "input_transform": transform_metadata,
            "validation": validation_metadata,
            "evaluation_sources": _mapping(config, "evaluation").get("sources", ()),
        }
        model = build_ds1_model(
            model_config,
            dropout_rng=backend.random_stream("dropout"),
        )
        objective = build_ds1_objective(objective_config, model.backend)
        if any(isinstance(layer, BatchNormalization) for layer in model.children()):
            model.forward(x_train[:1])
        optimizer = build_ds1_optimizer(
            optimizer_config,
            training_parameters(model, objective),
        )
        checkpoint_config = _mapping(config, "checkpoint")
        config_digest = compute_checkpoint_identity_digest(config, checkpoint_config)
        max_updates = training_config.get("max_updates")
        trainer_holder: dict[str, ForwardTrainer] = {}
        checkpoint_manager_holder: dict[str, CheckpointManager] = {}
        evaluation_config = _mapping(config, "evaluation")
        schedule = _mapping(evaluation_config, "schedule")
        request_by_set = _evaluation_requests(
            evaluation=evaluation_config,
            x_train=x_train,
            t_train=t_train,
            x_valid=x_valid,
            t_valid=t_valid,
            x_test=x_test,
            t_test=t_test,
        )

        artifact_root = Path(str(context.metadata["artifact_root"]))
        records_sink = DS1Records()
        records_sink.bind_artifact_root(artifact_root)
        events = create_supervised_events(
            config=config,
            backend=backend,
            context=context,
            records_sink=records_sink,
            schedule=schedule,
            request_by_set=request_by_set,
            trainer_holder=trainer_holder,
            checkpoint_manager_holder=checkpoint_manager_holder,
            checkpoint_config=checkpoint_config,
        )
        trainer = ForwardTrainer(
            model,
            objective,
            optimizer,
            max_epochs=int(training_config.get("max_epochs", 1)),
            max_updates=None if max_updates is None else int(max_updates),
            batch_size=int(loader_config.get("batch_size", 32)),
            drop_last=bool(loader_config.get("drop_last", False)),
            sampling_method=str(
                loader_config.get("sampling_method", "permutation_per_epoch")
            ),
            event_receivers=[events],
            batch_rng=backend.random_stream("batch_order"),
        )
        trainer_holder["trainer"] = trainer
        checkpoint_manager = setup_checkpoint_manager(
            checkpoint_root=Path(str(context.metadata["checkpoint_root"])),
            checkpoint_config=checkpoint_config,
            config_digest=config_digest,
            model=model,
            objective=objective,
            optimizer=optimizer,
            trainer=trainer,
        )
        checkpoint_manager_holder["manager"] = checkpoint_manager

        progress = context.metadata.get("progress_reporter")
        if progress is not None:
            progress.set_total_updates(
                trainer.planned_total_updates(len(x_train)),
                completed=trainer.global_step,
            )
        seed_batch_order(backend, streams)
        monitor = create_runtime_monitor(backend, _mapping(config, "profiling"))
        with training_summary(monitor):
            records = events.run(
                lambda: trainer.fit(x_train, t_train),
                start_update=trainer.global_step + 1,
            )
        record_manager_checkpoints(checkpoint_manager, records)
        latest = checkpoint_manager.current("latest")
        gap_metrics = {}
        if is_target_run(config):
            if latest is None:
                raise ValueError(
                    "full-train gap evaluation requires a latest checkpoint"
                )
            final_train, final_test, gap_metrics = evaluate_checkpoint_gap(
                trainer=trainer,
                model=model,
                checkpoint=latest.path,
                x_train=x_train,
                t_train=t_train,
                x_test=x_test,
                t_test=t_test,
            )
        else:
            final_train = trainer.evaluate(x_train, t_train)
            final_test = trainer.evaluate(x_test, t_test)
        profiling = monitor.metrics()
        metrics = build_final_metrics(
            train_loss=final_train.loss,
            test_loss=final_test.loss,
            train_accuracy=final_train.accuracy,
            test_accuracy=final_test.accuracy,
            profiling_metrics=profiling,
            total_updates=trainer.global_step,
            completed_epochs=trainer.epoch,
            samples_seen=sum(int(row["batch_size"]) for row in records.updates),
        )
        metrics.update(gap_metrics)
        return ExperimentResult(
            metrics=metrics,
            artifact_root=_artifact_root(context),
            model=model,
            metric_rows=records.mlflow_metric_rows(),
            profiling_metrics=profiling,
        )


_EXECUTORS = {
    "supervised_classification": SupervisedClassificationExecutor(),
}


def get_executor(kind: str):
    try:
        return _EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown DS1 implemented experiment kind: {kind}") from exc
