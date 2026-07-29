"""Portable PTB word2vec/RNNLM and character seq2seq experiment executors."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from mlprosection import Tensor
from mlprosection.datasets import load_ptb, load_sequence
from mlprosection.nn.sampling import UnigramSampler
from mlprosection.nn.model.architecture import (
    AttentionSeq2seq,
    BetterRnnlm,
    CBOW,
    CBOWBatchAdapter,
    PeekySeq2seq,
    Rnnlm,
    Seq2seq,
    SkipGram,
    SkipGramBatchAdapter,
    VanillaRnnlm,
)
from mlprosection.nn.objective import (
    FullSoftmax,
    NegativeSampling,
    TemporalSoftmaxCrossEntropy,
)
from mlprosection.optim.SGD import Adam, SGD
from mlprosection.optim.transform import ClipGradNorm
from mlprosection.trainer import (
    LanguageModelTrainer,
    Seq2seqTrainer,
    Word2VecTrainer,
)

from mlprosection.experiment.contracts import ExperimentResult
from mlprosection.experiment.checkpoint import (
    CheckpointManager,
    CheckpointRetentionPolicy,
    load_epoch_checkpoint,
    resolve_checkpoint_path,
)
from mlprosection.experiment.event_executor import EvaluationRequest, EventExperimentExecutor
from mlprosection.experiment.executor import ExperimentContext
from mlprosection.experiment.profiling import create_runtime_monitor, training_summary
from mlprosection.experiment.registry import register_executor
from mlprosection.experiment.reproducibility import configure_runtime, seed_batch_order
from mlprosection.profiling.backend import create_device_timer

from .records import DS2Records


def get_observation_executor(config: dict[str, object]):
    group_id = str(config.get("execution_group_id", ""))
    if group_id == "GO01":
        return AttentionAlignmentObservationExecutor()
    raise ValueError(f"unknown DS2 observation group: {group_id}")


def _mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _artifact_root(config: dict[str, object], context: ExperimentContext | None = None) -> Path:
    """Return this run's artifact root, never the legacy global results root."""
    if context is not None and (root := context.metadata.get("artifact_root")) is not None:
        return Path(str(root))
    return Path(str(_mapping(config, "run").get("artifact_root", "exp/ds2/results"))) / str(config["atomic_run_id"])


def _optimizer(config: dict[str, object], model, objective):
    values = _mapping(config, "optimizer")
    name = str(values.get("name", "adam"))
    params = [
        *((f"model.{name}", parameter) for name, parameter in model.named_parameters()),
        *((f"objective.{name}", parameter) for name, parameter in objective.named_parameters()),
    ]
    default_max_grad = {
        "language_modeling": 0.25,
        "seq2seq": 5.0,
    }.get(str(config.get("kind")))
    max_grad = _optional_max_grad(config, default=default_max_grad)
    hooks = None if max_grad is None else [ClipGradNorm(max_grad)]
    if name == "adam":
        return Adam(
            params,
            lr=float(values.get("learning_rate", 0.001)),
            pre_step_hooks=hooks,
        )
    if name == "sgd":
        return SGD(
            params,
            lr=float(values.get("learning_rate", 1.0)),
            pre_step_hooks=hooks,
        )
    raise ValueError(f"unsupported sequence optimizer: {name}")


def _apply_validation_decay(config: dict[str, object], optimizer) -> None:
    scheduler = _mapping(config, "scheduler")
    if str(scheduler.get("name", "constant")) == "validation_decay":
        optimizer.lr /= float(scheduler.get("factor", 4.0))


def _final(*, updates: int, epochs: int, samples: int, **values: float) -> dict[str, float]:
    return {
        "final/status/success": 1.0,
        "final/status/nan_detected": 0.0,
        "final/status/inf_detected": 0.0,
        "final/status/diverged": 0.0,
        "final/system/total_updates": float(updates),
        "final/system/completed_epochs": float(epochs),
        "final/system/samples_seen": float(samples),
        **{key: float(value) for key, value in values.items()},
    }


def _config_digest(config: dict[str, object]) -> str:
    checkpoint_config = dict(_mapping(config, "checkpoint"))
    checkpoint_config.pop("resume", None)
    identity = dict(config)
    identity["checkpoint"] = checkpoint_config
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _source_curve_from_objective(config: dict[str, object]):
    """Reduce pre-update source objectives at the book's zero-based cadence.

    The accumulator and completed point stay on the active backend until the
    record sink's bulk flush, so producing a point does not synchronize here.
    """
    recording = _mapping(config, "recording")
    curve = recording.get("source_curve", {})
    if not isinstance(curve, dict):
        return lambda _event: None
    every = int(curve.get("every_updates", 0))
    if every < 1:
        return lambda _event: None
    kind = str(curve.get("kind", "interval_mean_loss"))
    reducer = str(curve.get("reducer", "mean"))
    metric = "perplexity" if kind == "train_perplexity" else "loss"
    unit = "token" if kind == "train_perplexity" else "example"
    total = None
    book_total = None
    count = 0
    unit_count = 0
    update_start = None
    epoch_start = None
    plot_index = 0

    def reduce(event):
        nonlocal total, book_total, count, unit_count, update_start, epoch_start, plot_index
        if update_start is None:
            update_start = event.update
            epoch_start = event.epoch
        weight = int(event.unit_count) if reducer in {"token_weighted_mean", "exp_token_weighted_mean"} else 1
        weighted_objective = event.objective.data * weight
        total = weighted_objective if total is None else total + weighted_objective
        if event.book_objective is not None:
            weighted_book = event.book_objective.data * weight
            book_total = (
                weighted_book
                if book_total is None
                else book_total + weighted_book
            )
        count += 1
        unit_count += int(event.unit_count)
        if event.local_iteration % every != 0:
            return None
        denominator = unit_count if reducer in {"token_weighted_mean", "exp_token_weighted_mean"} else count
        value = Tensor(total / denominator, backend=event.objective.backend)
        if kind == "train_perplexity":
            value = Tensor(
                event.objective.backend.xp.exp(value.data),
                backend=event.objective.backend,
            )
        point = {
            "series_id": kind,
            "plot_index": plot_index,
            "update_start": update_start,
            "update_end": event.update,
            "epoch_start": epoch_start,
            "epoch_end": event.epoch,
            "unit": unit,
            "unit_count": unit_count,
            "metric": metric,
            "reducer": reducer,
            "value": value,
        }
        if book_total is not None:
            point["book_value"] = Tensor(
                book_total / denominator,
                backend=event.book_objective.backend,
            )
        total, book_total, count, unit_count = None, None, 0, 0
        update_start, epoch_start = None, None
        plot_index += 1
        return point

    return reduce


@register_executor("word2vec")
class Word2VecExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset, model_config, objective_config, training = (_mapping(config, key) for key in ("dataset", "model", "objective", "training"))
        corpus, word_to_id = _word2vec_corpus(_mapping(config, "dataset"))
        window = int(dataset.get("window_size", 5))
        contexts, targets = _contexts_targets(corpus, window)
        context.metadata["data"] = {
            "dataset_checksum": _array_digest(corpus),
            "split_checksum": _array_digest(contexts, targets),
        }
        objective_name = str(objective_config.get("name", "NegativeSampling"))
        sampler = None
        if objective_name == "NegativeSampling":
            sampler_values = _mapping(objective_config, "sampler")
            sampler = UnigramSampler.from_corpus(
                corpus,
                vocab_size=len(word_to_id),
                backend=backend,
                power=float(sampler_values.get("power", 0.75)),
                rejection_rounds=int(sampler_values.get("rejection_rounds", 4)),
                algorithm=str(
                    sampler_values.get(
                        "algorithm", UnigramSampler.ALIAS_REJECTION
                    )
                ),
            )
            context.metadata["negative_sampler"] = sampler.metadata
        architecture = str(model_config.get("name", "CBOW"))
        model_type, adapter = {
            "CBOW": (CBOW, CBOWBatchAdapter()),
            "SkipGram": (SkipGram, SkipGramBatchAdapter()),
        }.get(architecture, (None, None))
        if model_type is None:
            raise ValueError(f"unknown Word2Vec model name: {architecture}")
        embedding_size = int(model_config.get("embedding_size", 100))
        model = model_type(
            len(word_to_id), embedding_size, backend=backend
        )
        if objective_name == "FullSoftmax":
            objective = FullSoftmax(
                len(word_to_id), embedding_size,
                reduction=str(objective_config.get("reduction", "mean")),
                backend=backend,
            )
        elif objective_name == "NegativeSampling":
            objective = NegativeSampling(
                len(word_to_id), embedding_size,
                negative_samples=int(objective_config.get("negative_samples", 5)),
                reduction=str(objective_config.get("reduction", "mean")),
                sampler=sampler,
                backend=backend,
            )
        else:
            raise ValueError(f"unknown Word2Vec objective name: {objective_name}")
        optimizer = _optimizer(config, model, objective)
        seed_batch_order(backend, streams)
        loader = _mapping(config, "loader")
        batch_size, epochs = int(loader.get("batch_size", 100)), int(training.get("max_epochs", 10))
        max_updates = training.get("max_updates")
        x = backend.xp.asarray(contexts, dtype=backend.xp.int64)
        t = backend.xp.asarray(targets, dtype=backend.xp.int64)
        artifact_root = _artifact_root(config, context)
        records_sink = DS2Records()
        records_sink.bind_artifact_root(artifact_root)
        monitor = create_runtime_monitor(backend, _mapping(config, "profiling"))
        trainer = Word2VecTrainer(
            model, objective, optimizer, batch_adapter=adapter,
            max_epochs=epochs, batch_size=batch_size,
            max_updates=None if max_updates is None else int(max_updates),
            drop_last=bool(loader.get("drop_last", True)),
            event_receivers=[],
        )
        checkpoint_manager = _checkpoint_manager(
            config, context, model=model, objective=objective, optimizer=optimizer, trainer=trainer,
        )
        controller = EventExperimentExecutor(
            records=records_sink, evaluate=lambda _request: None,
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
        with training_summary(monitor):
            records = controller.run(
                lambda: trainer.fit(Tensor(x, backend=backend), Tensor(t, backend=backend)),
                start_update=trainer.global_step + 1,
            )
        _record_retained_checkpoints(records, checkpoint_manager)
        records.flush()
        final_loss = _recorded_float(records.updates[-1]["loss"]) if records.updates else 0.0
        final_metrics = {"final/train/loss": final_loss}
        if records.updates and records.updates[-1].get("book_loss") is not None:
            final_metrics["final/train/book_loss"] = _recorded_float(
                records.updates[-1]["book_loss"]
            )
        profiling_metrics = monitor.metrics()
        return ExperimentResult(
            metrics=_final(updates=trainer.global_step, epochs=trainer.epoch, samples=len(x) * trainer.epoch, **final_metrics),
            artifact_root=artifact_root, model=model, metric_rows=records.mlflow_metric_rows(),
            profiling_metrics=profiling_metrics,
        )


@register_executor("language_modeling")
class LanguageModelExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        ptb = load_ptb()
        model_config, loader, training = (_mapping(config, key) for key in ("model", "loader", "training"))
        dataset, evaluation = _mapping(config, "dataset"), _mapping(config, "evaluation")
        train_corpus, vocab_size = _language_model_training_corpus(ptb["train"], dataset)
        model = _language_model(str(model_config.get("name")), vocab_size, model_config, backend)
        objective = TemporalSoftmaxCrossEntropy(
            reduction=str(_mapping(config, "objective").get("reduction", "mean")),
            backend=backend,
        )
        optimizer = _optimizer(config, model, objective)
        seed_batch_order(backend, streams)
        train = Tensor(backend.xp.asarray(train_corpus[:-1], dtype=backend.xp.int64), backend=backend)
        train_targets = Tensor(backend.xp.asarray(train_corpus[1:], dtype=backend.xp.int64), backend=backend)
        max_epochs = int(training.get("max_epochs", 4))
        max_updates = training.get("max_updates")
        valid = Tensor(backend.xp.asarray(ptb["valid"][:-1], dtype=backend.xp.int64), backend=backend)
        valid_targets = Tensor(backend.xp.asarray(ptb["valid"][1:], dtype=backend.xp.int64), backend=backend)
        test = Tensor(backend.xp.asarray(ptb["test"][:-1], dtype=backend.xp.int64), backend=backend)
        test_targets = Tensor(backend.xp.asarray(ptb["test"][1:], dtype=backend.xp.int64), backend=backend)
        context.metadata["data"] = {
            "dataset_checksum": _array_digest(ptb["train"], ptb["valid"], ptb["test"]),
            "split_checksum": _array_digest(train_corpus, ptb["valid"], ptb["test"]),
        }
        valid_ppl = float("inf")
        test_ppl = float("inf")
        best_valid = float("inf")
        best_valid_epoch = 0
        selected_checkpoint_path: Path | None = None
        config_digest = _config_digest(config)
        valid_every_epochs = int(evaluation.get("valid_every_epochs", 1))
        test_every_epochs = int(evaluation.get("test_every_epochs", 1))
        test_at_end = bool(evaluation.get("test_at_end", False))
        requests = {
            "valid": EvaluationRequest("ptb-valid", "valid", (valid, valid_targets), ("perplexity",)),
            "test": EvaluationRequest("ptb-test", "test", (test, test_targets), ("perplexity",)),
        }
        trainer = LanguageModelTrainer(
            model, objective, optimizer, max_epochs=max_epochs,
            batch_size=int(loader.get("batch_size", 20)), time_size=int(loader.get("time_size", 35)),
            max_updates=None if max_updates is None else int(max_updates),
        )
        checkpoint_manager = _checkpoint_manager(
            config, context, model=model, objective=objective, optimizer=optimizer, trainer=trainer,
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
            nonlocal valid_ppl, test_ppl, best_valid, best_valid_epoch, selected_checkpoint_path
            if request.split == "valid":
                valid_ppl = float(result.perplexity)
                if valid_ppl < best_valid:
                    best_valid, best_valid_epoch = valid_ppl, step
                    if bool(_mapping(config, "checkpoint").get("save_best", False)):
                        records_sink.flush()
                        selected_checkpoint_path = checkpoint_manager.save_best().path
                else:
                    _apply_validation_decay(config, optimizer)
            else:
                test_ppl = float(result.perplexity)
        controller = EventExperimentExecutor(
            records=records_sink, evaluate=evaluate_request,
            epoch_requests=epoch_requests, after_evaluation=after_evaluation,
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
            if bool(_mapping(config, "checkpoint").get("save_best", False)):
                if selected_checkpoint_path is None:
                    raise RuntimeError("selected checkpoint is required before terminal test evaluation")
                load_epoch_checkpoint(path=selected_checkpoint_path, model=model, objective=objective, optimizer=optimizer, trainer=trainer, config_digest=config_digest)
            test_ppl = float(trainer.evaluate(test, test_targets).perplexity)
        _record_retained_checkpoints(
            records, checkpoint_manager,
            best_metric="valid/perplexity",
            best_value=None if best_valid == float("inf") else best_valid,
        )
        records.flush()
        final_train_ppl = _backend_exp_float(backend, records.updates[-1]["loss"]) if records.updates else float("inf")
        final_metrics = {
            "final/train/perplexity": final_train_ppl,
            "final/train/ppl": final_train_ppl,
        }
        if test_ppl < float("inf"):
            final_metrics.update({
                "final/test/perplexity": test_ppl,
                "final/test/ppl": test_ppl,
            })
        if best_valid < float("inf"):
            final_metrics.update({
                "final/valid/perplexity": valid_ppl,
                "final/valid/ppl": valid_ppl,
                "final/best_valid_ppl": best_valid,
                "final/best_valid_epoch": float(best_valid_epoch),
            })
        profiling_metrics = monitor.metrics()
        return ExperimentResult(
            metrics=_final(updates=trainer.global_step, epochs=trainer.epoch, samples=len(train) * trainer.epoch, **final_metrics),
            artifact_root=artifact_root,
            model=model,
            metric_rows=records.mlflow_metric_rows(),
            profiling_metrics=profiling_metrics,
        )


@register_executor("seq2seq")
class Seq2SeqExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset, model_config, loader, training = (_mapping(config, key) for key in ("dataset", "model", "loader", "training"))
        split_seed = int(dataset.get("split_seed", streams.dataset_split))
        split_algorithm = str(dataset.get("split_algorithm", "default_rng"))
        data = load_sequence(
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
            "dataset_checksum": _file_digest(_sequence_dataset_path(str(dataset["file"]))),
            "split_checksum": _array_digest(x_train, t_train, x_test, t_test),
        }
        model = _seq_model(str(model_config.get("name")), len(data["char_to_id"]), model_config, backend)
        objective = TemporalSoftmaxCrossEntropy(
            reduction=str(_mapping(config, "objective").get("reduction", "mean")),
            backend=backend,
        )
        optimizer = _optimizer(config, model, objective)
        seed_batch_order(backend, streams)
        batch_size, epochs = int(loader.get("batch_size", 128)), int(training.get("max_epochs", 10))
        max_updates = training.get("max_updates")
        train_x = Tensor(backend.xp.asarray(x_train, dtype=backend.xp.int64), backend=backend)
        train_t = Tensor(backend.xp.asarray(t_train, dtype=backend.xp.int64), backend=backend)
        test_source = (Tensor(backend.xp.asarray(x_test, dtype=backend.xp.int64), backend=backend), Tensor(backend.xp.asarray(t_test, dtype=backend.xp.int64), backend=backend))
        request = EvaluationRequest("sequence-test-full", "test", test_source, ("exact_match_accuracy", "token_accuracy"))
        trainer = Seq2seqTrainer(
            model, objective, optimizer, max_epochs=epochs, batch_size=batch_size,
            start_id=data["char_to_id"]["_"],
            max_updates=None if max_updates is None else int(max_updates),
            drop_last=bool(loader.get("drop_last", False)),
        )
        artifact_root = _artifact_root(config, context)
        records = DS2Records()
        records.bind_artifact_root(artifact_root)
        monitor = create_runtime_monitor(backend, _mapping(config, "profiling"))
        checkpoint_config = _mapping(config, "checkpoint")
        checkpoint_manager = _checkpoint_manager(
            config, context, model=model, objective=objective, optimizer=optimizer, trainer=trainer,
        )
        best_exact = -1.0

        def after_evaluation(_request, result, axis, step):
            nonlocal best_exact
            if axis == "epoch" and result.exact_match_accuracy is not None:
                records.add_source_curve({
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
                })
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
                if bool(checkpoint_config.get("save_best", False)) and result.exact_match_accuracy > best_exact:
                    best_exact = float(result.exact_match_accuracy)
                    records.flush()
                    checkpoint_manager.save_best()

        controller = EventExperimentExecutor(
            records=records,
            evaluate=lambda _request: trainer.evaluate(*test_source, metrics=("exact_match_accuracy", "token_accuracy")),
            epoch_requests=lambda _event: (request,),
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
            records, checkpoint_manager,
            best_metric="test/exact_match_accuracy",
            best_value=None if best_exact < 0 else best_exact,
        )
        selected = checkpoint_manager.current("best")
        record_checkpoint = context.metadata.get("record_eval_checkpoint")
        if selected is not None and callable(record_checkpoint):
            record_checkpoint(selected.path)
        records.flush()
        last_evaluation = records.evaluations[-2:] if len(records.evaluations) >= 2 else []
        final_values = {"final/train/loss": _recorded_float(records.updates[-1]["loss"]) if records.updates else 0.0}
        for row in last_evaluation:
            final_values[f"final/test/{row['metric'].replace('_accuracy', '')}"] = float(row["value"])
        return ExperimentResult(
            metrics=_final(updates=trainer.global_step, epochs=trainer.epoch, samples=len(x_train) * trainer.epoch, **final_values),
            artifact_root=artifact_root,
            model=model,
            metric_rows=records.mlflow_metric_rows(),
            profiling_metrics=monitor.metrics(),
        )


class AttentionAlignmentObservationExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset = _mapping(config, "dataset")
        model_config = _mapping(config, "model")
        checkpoint_config = _mapping(config, "checkpoint")
        checkpoint_path = checkpoint_config.get("source_path") or checkpoint_config.get("source_checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("DS2 GO01 requires checkpoint.source_path")
        checkpoint_path = resolve_checkpoint_path(Path(str(checkpoint_path)))
        data = load_sequence(
            str(dataset["file"]),
            seed=int(dataset.get("split_seed", 1984)),
            split_algorithm=str(dataset.get("split_algorithm", "legacy_numpy_randomstate")),
        )
        x_test, t_test = data["test"]
        if bool(dataset.get("reverse", True)):
            x_test = x_test[:, ::-1]
        context.metadata["data"] = {
            "split_seed": int(dataset.get("split_seed", 1984)),
            "split_algorithm": str(dataset.get("split_algorithm", "legacy_numpy_randomstate")),
            "dataset_checksum": _file_digest(_sequence_dataset_path(str(dataset["file"]))),
            "observation_split_checksum": _array_digest(x_test, t_test),
        }
        model = _seq_model(str(model_config.get("name", "AttentionSeq2seq")), len(data["char_to_id"]), model_config, backend)
        if not isinstance(model, AttentionSeq2seq):
            raise ValueError("DS2 GO01 requires AttentionSeq2seq model")
        _load_model_checkpoint(model, Path(str(checkpoint_path)))
        artifact_root = _artifact_root(config, context)
        records = DS2Records()
        records.bind_artifact_root(artifact_root)
        recording = _mapping(config, "recording")
        attention_config = _mapping(recording, "attention")
        count = int(attention_config.get("count", 5))
        example_ids = _attention_example_ids(
            size=len(x_test),
            count=count,
            seed=int(attention_config.get("selection_seed", 1984)),
        )
        start_id = data["char_to_id"]["_"]
        id_to_char = data["id_to_char"]
        render_examples = []
        for example_id in example_ids:
            question = Tensor(backend.xp.asarray(x_test[example_id:example_id + 1], dtype=backend.xp.int64), backend=backend)
            expected = [int(value) for value in t_test[example_id][1:]]
            predicted, weights = _generate_attention_with_weights(model, question, start_id, len(expected), backend)
            source_text = _decode_ids(x_test[example_id], id_to_char)
            target_text = _decode_ids(expected, id_to_char)
            prediction_text = _decode_ids(predicted, id_to_char)
            records.add_prediction({
                "epoch": 0,
                "example_id": example_id,
                "source": source_text,
                "target": target_text,
                "prediction": prediction_text,
                "exact_match": int(predicted == expected),
                "token_correct": sum(left == right for left, right in zip(predicted, expected, strict=True)),
                "token_count": len(expected),
            })
            render_examples.append({"example_id": example_id, "source": source_text, "target": target_text, "prediction": prediction_text})
            for decode_step in range(weights.shape[0]):
                for encoder_position in range(weights.shape[1]):
                    records.add_attention({
                        "example_id": example_id,
                        "decode_step": decode_step,
                        "encoder_position": encoder_position,
                        "weight": float(weights[decode_step, encoder_position]),
                    })
        records.set_attention_render({
            "source_checkpoint": str(Path(str(checkpoint_path))),
            "source_checkpoint_sha256": _path_digest(Path(str(checkpoint_path))),
            "example_selection_seed": int(attention_config.get("selection_seed", 1984)),
            "decode_policy": str(attention_config.get("decode", "greedy")),
            "input_reversal": bool(dataset.get("reverse", True)),
            "x_axis": "encoder_position",
            "y_axis": "decode_step",
            "y_axis_inverted": True,
            "color_range": [0.0, 1.0],
            "examples": render_examples,
        })
        records.flush()
        return ExperimentResult(
            metrics={"final/status/success": 1.0, "final/system/total_updates": 0.0, "final/system/completed_epochs": 0.0, "final/system/samples_seen": float(len(example_ids))},
            artifact_root=artifact_root,
            model=model,
            metric_rows=(),
            profiling_metrics={},
        )


def _device_timer(config: dict[str, object], backend):
    profiling = _mapping(config, "profiling")
    return create_device_timer(backend, enabled=bool(profiling.get("device_timing", False)))


def _checkpoint_manager(config, context, *, model, objective, optimizer, trainer) -> CheckpointManager:
    checkpoint = _mapping(config, "checkpoint")
    return CheckpointManager(
        root=Path(str(context.metadata["checkpoint_root"])),
        model=model,
        objective=objective,
        optimizer=optimizer,
        trainer=trainer,
        config_digest=_config_digest(config),
        policy=CheckpointRetentionPolicy.from_mapping(checkpoint),
    )


def _save_epoch_roles(manager: CheckpointManager) -> None:
    manager.save_latest()
    manager.save_periodic_if_due()


def _record_retained_checkpoints(
    records: DS2Records,
    manager: CheckpointManager,
    *,
    best_metric: str = "",
    best_value: float | None = None,
) -> None:
    latest = manager.current("latest")
    if latest is not None:
        records.add_checkpoint(
            update=latest.update, epoch=latest.epoch, kind="latest",
            path=latest.path, sha256=latest.sha256,
            checkpoint_id=f"latest-epoch-{latest.epoch:04d}",
        )
    best = manager.current("best")
    if best is not None:
        records.add_checkpoint(
            update=best.update, epoch=best.epoch, kind="selected",
            path=best.path, sha256=best.sha256,
            checkpoint_id=f"selected-epoch-{best.epoch:04d}",
            selection_metric=best_metric,
            selection_value="" if best_value is None else best_value,
        )
    for periodic in manager.retained_periodic():
        records.add_checkpoint(
            update=periodic.update, epoch=periodic.epoch, kind="periodic",
            path=periodic.path, sha256=periodic.sha256,
            checkpoint_id=f"periodic-epoch-{periodic.epoch:04d}",
        )


def _contexts_targets(corpus, window: int):
    width = 2 * window + 1
    windows = np.lib.stride_tricks.sliding_window_view(corpus, width)
    contexts = np.concatenate((windows[:, :window], windows[:, window + 1:]), axis=1)
    return contexts, corpus[window:-window]


def _language_model_training_corpus(corpus, dataset: dict[str, object]):
    """Resolve the training slice and its source-compatible vocabulary size."""
    train_limit = int(dataset.get("train_limit", len(corpus)))
    train_corpus = corpus[:train_limit]
    if len(train_corpus) < 2:
        raise ValueError("language-model corpus must contain at least two tokens")
    return train_corpus, int(np.max(train_corpus)) + 1


def _word2vec_corpus(dataset: dict[str, object]):
    """Load PTB or the book's fixed toy sentence for Word2Vec experiments."""
    if str(dataset.get("id")) != "DS-TOY-W2V":
        ptb = load_ptb()
        return ptb["train"], ptb["word_to_id"]
    text = str(dataset.get("text", "You say goodbye and I say hello."))
    words = text.lower().replace(".", " .").split()
    word_to_id = {word: index for index, word in enumerate(dict.fromkeys(words))}
    return np.asarray([word_to_id[word] for word in words], dtype=np.int64), word_to_id


def _optional_max_grad(config: dict[str, object], *, default: float | None = None) -> float | None:
    value = _mapping(config, "policy").get("max_grad", default)
    return None if value is None else float(value)


def _record_seq_predictions(
    records: DS2Records,
    model,
    questions,
    answers,
    char_to_id,
    id_to_char,
    backend,
    recording: dict[str, object],
    *,
    epoch: int,
    predictions=None,
) -> None:
    config = recording.get("predictions")
    if not isinstance(config, dict):
        return
    if str(config.get("split", "test")) != "test":
        raise ValueError("seq2seq predictions currently support split: test")
    count = min(int(config.get("count", 10)), len(questions))
    start_id = char_to_id["_"]
    was_training = bool(getattr(model, "training", True))
    model.train(False)
    try:
        for example_id in range(count):
            expected = [int(value) for value in answers[example_id][1:]]
            if predictions is None:
                question = Tensor(
                    backend.xp.asarray(
                        questions[example_id:example_id + 1],
                        dtype=backend.xp.int64,
                    ),
                    backend=backend,
                )
                predicted = model.generate(question, start_id, len(expected))
            else:
                predicted = [int(value) for value in predictions[example_id]]
            records.add_prediction({
                "epoch": epoch,
                "example_id": example_id,
                "source": _decode_ids(questions[example_id], id_to_char),
                "target": _decode_ids(expected, id_to_char),
                "prediction": _decode_ids(predicted, id_to_char),
                "exact_match": int(predicted == expected),
                "token_correct": sum(left == right for left, right in zip(predicted, expected, strict=True)),
                "token_count": len(expected),
            })
    finally:
        model.train(was_training)


def _generate_attention_with_weights(model: AttentionSeq2seq, question: Tensor, start_id: int, sample_size: int, backend) -> tuple[list[int], np.ndarray]:
    xp = backend.xp
    was_training = bool(getattr(model, "training", True))
    model.train(False)
    try:
        enc_hs = model.encoder.forward(question)
        model.decoder.lstm.set_state(enc_hs[:, -1, :])
        sample_id = xp.asarray(start_id, dtype=xp.int64)
        sampled = []
        weights = []
        for _ in range(sample_size):
            out = model.decoder.embed.forward(Tensor(sample_id.reshape((1, 1)), backend=backend))
            dec_hs = model.decoder.lstm.forward(out)
            context = model.decoder.attention.forward(enc_hs, dec_hs)
            weights.append(model.decoder.attention.weights[0, 0].copy())
            score = model.decoder.affine.forward(Tensor(xp.concatenate((context.data, dec_hs.data), axis=2), backend=backend))
            sample_id = score.data.reshape(-1).argmax()
            sampled.append(sample_id)
        host_ids = backend.to_numpy(xp.stack(sampled)) if sampled else np.asarray([], dtype=np.int64)
        host_weights = backend.to_numpy(xp.stack(weights)) if weights else np.empty((0, 0))
        return [int(value) for value in host_ids], np.asarray(host_weights)
    finally:
        model.train(was_training)


def _decode_ids(values, id_to_char: dict[int, str]) -> str:
    return "".join(id_to_char[int(value)] for value in values)


def _recorded_float(value: object) -> float:
    if hasattr(value, "backend") and hasattr(value, "data"):
        return value.backend.scalar_to_float(value.data)
    return float(value)


def _backend_exp_float(backend, value: object) -> float:
    result = backend.xp.exp(backend.xp.asarray(_recorded_float(value)))
    return backend.scalar_to_float(result)


def _attention_example_ids(*, size: int, count: int, seed: int) -> list[int]:
    if size < 1:
        return []
    rng = np.random.RandomState(seed)
    return [int(rng.randint(0, size)) for _ in range(count)]


def _sequence_dataset_path(file_name: str) -> Path:
    from mlprosection.datasets import sequence

    return Path(sequence.__file__).resolve().parent / file_name


def _array_digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _load_model_checkpoint(model, path: Path) -> None:
    path = resolve_checkpoint_path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    manifest = json.loads(
        (path / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 2:
        raise ValueError("only checkpoint schema version 2 is supported")
    model.load_params_npz(path / "model_parameters.npz")


def _path_digest(path: Path) -> str:
    if path.is_file():
        return _file_digest(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(_file_digest(child).encode())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _language_model(name: str, vocab_size: int, values: dict[str, object], backend):
    kwargs = {"vocab_size": vocab_size, "wordvec_size": int(values.get("wordvec_size", 100)), "hidden_size": int(values.get("hidden_size", 100)), "backend": backend}
    if name == "VanillaRnnlm":
        return VanillaRnnlm(**kwargs)
    if name == "Rnnlm":
        return Rnnlm(**kwargs)
    if name == "BetterRnnlm":
        return BetterRnnlm(**kwargs, dropout_ratio=float(values.get("dropout_ratio", 0.5)))
    raise ValueError(f"unknown language-model name: {name}")


def _seq_model(name: str, vocab_size: int, values: dict[str, object], backend):
    kwargs = {"vocab_size": vocab_size, "wordvec_size": int(values.get("wordvec_size", 16)), "hidden_size": int(values.get("hidden_size", 128)), "backend": backend}
    if name == "Seq2seq":
        return Seq2seq(**kwargs)
    if name == "PeekySeq2seq":
        return PeekySeq2seq(**kwargs)
    if name == "AttentionSeq2seq":
        return AttentionSeq2seq(**kwargs)
    raise ValueError(f"unknown seq2seq name: {name}")


def _seq_accuracy(model, questions, answers, char_to_id, backend) -> tuple[float, float]:
    start_id = char_to_id["_"]
    exact = 0
    token_correct = 0
    token_total = 0
    model.eval()
    for question, answer in zip(questions, answers, strict=True):
        predicted = model.generate(Tensor(backend.xp.asarray(question[None, :], dtype=backend.xp.int64), backend=backend), start_id, len(answer) - 1)
        expected = [int(value) for value in answer[1:]]
        exact += int(predicted == expected)
        token_correct += sum(left == right for left, right in zip(predicted, expected, strict=True))
        token_total += len(expected)
    model.train(True)
    return exact / len(questions), token_correct / max(token_total, 1)


def _save_attention_artifact(model, questions, answers, backend, context: ExperimentContext) -> float | None:
    attention = getattr(getattr(model, "decoder", None), "attention", None)
    if attention is None:
        return None
    model.eval()
    question = Tensor(backend.xp.asarray(questions[:1], dtype=backend.xp.int64), backend=backend)
    answer = Tensor(backend.xp.asarray(answers[:1, :-1], dtype=backend.xp.int64), backend=backend)
    encoder_states = model.encoder.forward(question)
    model.decoder.forward(answer, encoder_states)
    weights = attention.weights
    if weights is None:
        return None
    values = backend.to_numpy(weights[0])
    entropy = float(-(values * np.log(values + 1e-12)).sum(axis=1).mean())
    root = Path(str(context.metadata.get("artifact_root", "exp/ds2/results/runs")))
    path = root / "analysis" / "attention_map.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, attention=values, entropy=entropy)
    model.train(True)
    return entropy
