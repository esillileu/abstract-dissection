"""Portable PTB word2vec/RNNLM and character seq2seq experiment executors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from deepscratch.core import Tensor, configure_runtime, seed_batch_order
from deepscratch.nn.model.architecture import AttentionSeq2seq
from deepscratch.profiling import create_runtime_monitor, training_summary
from deepscratch.profiling.backend import create_device_timer
from deepscratch.trainer import (
    FusedNegativeSamplingTrainer,
    LanguageModelTrainer,
    Seq2seqTrainer,
    Word2VecTrainer,
)

from dlfs.adapters.checkpoint import (
    create_deepscratch_checkpoint_manager,
    load_deepscratch_model_parameters,
)
from dlfs.ds2.implemented.adapters import (
    build_language_model,
    build_seq2seq_model,
    build_sequence_objective,
    build_sequence_optimizer,
    build_unigram_sampler,
    build_word2vec_batch_adapter,
    build_word2vec_model,
    build_word2vec_objective,
    contexts_targets,
    language_model_training_corpus,
    load_ds2_ptb,
    load_ds2_sequence,
    load_ds2_word2vec_corpus,
)
from dlfs.ds2.statistical import (
    create_cooccurrence_matrix,
    factorize_ppmi,
    positive_pmi,
)
from repro_core.context import ExperimentContext
from repro_core.context.checkpoint import (
    CheckpointManager,
    CheckpointRetentionPolicy,
    resolve_checkpoint_path,
)
from repro_core.context.contracts import ExperimentResult
from repro_core.context.event_executor import EvaluationRequest, EventExperimentExecutor
from repro_core.registry import register_executor

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


def _artifact_root(
    config: dict[str, object], context: ExperimentContext | None = None
) -> Path:
    """Return this run's artifact root, never the legacy global results root."""
    if (
        context is not None
        and (root := context.metadata.get("artifact_root")) is not None
    ):
        return Path(str(root))
    raise ValueError("experiment context is missing artifact_root")


def _publish_array_checkpoint(context: ExperimentContext, **arrays: np.ndarray) -> Path:
    """Publish analysis arrays through the canonical v2 checkpoint pointer."""
    import os

    root = Path(str(context.metadata["checkpoint_root"]))
    root.mkdir(parents=True, exist_ok=True)
    target = root / "final.npz"
    temporary = root / ".final.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    pointer = {
        "schema_version": 2,
        "role": "latest",
        "path": target.name,
        "sha256": digest,
        "epoch": 0,
        "update": 0,
    }
    temporary_pointer = root / ".latest.json.tmp"
    temporary_pointer.write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_pointer, root / "latest.json")
    return target


@register_executor("count_based_embedding")
class CountBasedEmbeddingExecutor:
    """Build and persist PTB PPMI representations and their factorizations."""

    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        _backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset = _mapping(config, "dataset")
        model = _mapping(config, "model")
        corpus, word_to_id = load_ds2_word2vec_corpus(dataset)
        window_size = int(dataset.get("window_size", 2))
        components = min(int(model.get("embedding_size", 100)), len(word_to_id))
        method = str(model.get("name", "ppmi")).lower()
        seed = int(config.get("seed", 1))

        started = perf_counter()
        cooccurrence = create_cooccurrence_matrix(corpus, len(word_to_id), window_size)
        cooccurrence_s = perf_counter() - started
        started = perf_counter()
        ppmi = positive_pmi(cooccurrence)
        ppmi_s = perf_counter() - started
        started = perf_counter()
        vectors, singular_values, right_factors = factorize_ppmi(
            ppmi,
            method=method,
            components=components,
            seed=seed,
            n_iter=int(model.get("n_iter", 5)),
        )
        decomposition_s = perf_counter() - started
        total_s = cooccurrence_s + ppmi_s + decomposition_s

        artifact_root = _artifact_root(config, context)
        artifact_root.mkdir(parents=True, exist_ok=True)
        matrix_path = artifact_root / "statistical_matrices.npz"
        payload = {
            "cooccurrence": cooccurrence,
            "ppmi": ppmi,
            "word_vectors": vectors,
            "singular_values": singular_values,
        }
        if right_factors is not None:
            payload["right_factors"] = right_factors
        np.savez_compressed(matrix_path, **payload)
        timing_path = artifact_root / "timing.json"
        timing = {
            "cooccurrence_s": cooccurrence_s,
            "ppmi_s": ppmi_s,
            "decomposition_s": decomposition_s,
            "total_s": total_s,
            "method": method,
        }
        timing_path.write_text(
            json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checkpoint = _publish_array_checkpoint(
            context,
            W_in=vectors,
            word_vectors=vectors,
            singular_values=singular_values,
        )
        metrics = _final(
            updates=0,
            epochs=0,
            samples=len(corpus),
            **{
                "final/runtime/cooccurrence_s": cooccurrence_s,
                "final/runtime/ppmi_s": ppmi_s,
                "final/runtime/decomposition_s": decomposition_s,
                "runtime/train_total_s": total_s,
            },
        )
        return ExperimentResult(
            metrics=metrics,
            artifact_root=artifact_root,
            artifacts=(matrix_path, timing_path, checkpoint),
            metric_rows=tuple(
                (0, name, value)
                for name, value in (
                    ("runtime/cooccurrence_s", cooccurrence_s),
                    ("runtime/ppmi_s", ppmi_s),
                    ("runtime/decomposition_s", decomposition_s),
                    ("runtime/total_s", total_s),
                )
            ),
        )


@register_executor("performance_profile")
class ProfileExecutor:
    """Dispatch canonical profile studies through their shared result contract."""

    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        from dlfs.ds2.profile.studies import resolve
        from dlfs.profile.result import to_experiment_result

        profiling = _mapping(config, "profiling")
        study = resolve(str(profiling.get("study_kind", "")))
        result = study.run(config, context)
        context.metadata["profile"] = {
            "study_id": result.study_id,
            "group_id": result.group_id,
            "study_kind": result.study_kind,
            "source_study": result.source_study,
        }
        return to_experiment_result(
            result,
            artifact_root=_artifact_root(config, context),
        )


def _apply_validation_decay(config: dict[str, object], optimizer) -> None:
    scheduler = _mapping(config, "scheduler")
    if str(scheduler.get("name", "constant")) == "validation_decay":
        optimizer.lr /= float(scheduler.get("factor", 4.0))


def _final(
    *, updates: int, epochs: int, samples: int, **values: float
) -> dict[str, float]:
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
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()


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
    reset_each_epoch = bool(
        _mapping(config, "policy").get("source_curve_reset_each_epoch", False)
    )
    active_epoch = None

    def reduce(event):
        nonlocal \
            total, \
            book_total, \
            count, \
            unit_count, \
            update_start, \
            epoch_start, \
            plot_index, \
            active_epoch
        if (
            reset_each_epoch
            and active_epoch is not None
            and event.epoch != active_epoch
        ):
            total, book_total, count, unit_count = None, None, 0, 0
            update_start, epoch_start = None, None
        active_epoch = event.epoch
        if update_start is None:
            update_start = event.update
            epoch_start = event.epoch
        weight = (
            int(event.unit_count)
            if reducer in {"token_weighted_mean", "exp_token_weighted_mean"}
            else 1
        )
        weighted_objective = event.objective.data * weight
        total = weighted_objective if total is None else total + weighted_objective
        if event.book_objective is not None:
            weighted_book = event.book_objective.data * weight
            book_total = (
                weighted_book if book_total is None else book_total + weighted_book
            )
        count += 1
        unit_count += int(event.unit_count)
        if event.local_iteration % every != 0:
            return None
        denominator = (
            unit_count
            if reducer in {"token_weighted_mean", "exp_token_weighted_mean"}
            else count
        )
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


@register_executor("language_modeling")
class LanguageModelExecutor:
    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        ptb = load_ds2_ptb()
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


@register_executor("seq2seq")
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
        data = load_ds2_sequence(
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


class AttentionAlignmentObservationExecutor:
    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset = _mapping(config, "dataset")
        model_config = _mapping(config, "model")
        checkpoint_config = _mapping(config, "checkpoint")
        checkpoint_path = checkpoint_config.get("source_path") or checkpoint_config.get(
            "source_checkpoint_path"
        )
        if checkpoint_path is None:
            raise ValueError("DS2 GO01 requires checkpoint.source_path")
        checkpoint_path = resolve_checkpoint_path(Path(str(checkpoint_path)))
        data = load_ds2_sequence(
            str(dataset["file"]),
            seed=int(dataset.get("split_seed", 1984)),
            split_algorithm=str(
                dataset.get("split_algorithm", "legacy_numpy_randomstate")
            ),
        )
        x_test, t_test = data["test"]
        if bool(dataset.get("reverse", True)):
            x_test = x_test[:, ::-1]
        context.metadata["data"] = {
            "split_seed": int(dataset.get("split_seed", 1984)),
            "split_algorithm": str(
                dataset.get("split_algorithm", "legacy_numpy_randomstate")
            ),
            "dataset_checksum": _file_digest(
                _sequence_dataset_path(str(dataset["file"]))
            ),
            "observation_split_checksum": _array_digest(x_test, t_test),
        }
        model = build_seq2seq_model(
            str(model_config.get("name", "AttentionSeq2seq")),
            len(data["char_to_id"]),
            model_config,
            backend,
        )
        if not isinstance(model, AttentionSeq2seq):
            raise ValueError("DS2 GO01 requires AttentionSeq2seq model")
        load_deepscratch_model_parameters(Path(str(checkpoint_path)), model)
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
            question = Tensor(
                backend.xp.asarray(
                    x_test[example_id : example_id + 1], dtype=backend.xp.int64
                ),
                backend=backend,
            )
            expected = [int(value) for value in t_test[example_id][1:]]
            conditioning = str(
                attention_config.get(
                    "conditioning", attention_config.get("decode", "greedy")
                )
            )
            if conditioning == "teacher_forcing":
                predicted, weights = _teacher_forced_attention_with_weights(
                    model,
                    question,
                    t_test[example_id],
                    backend,
                )
            elif conditioning == "greedy":
                predicted, weights = _generate_attention_with_weights(
                    model, question, start_id, len(expected), backend
                )
            else:
                raise ValueError(
                    "attention.conditioning must be teacher_forcing or greedy"
                )
            weights = weights[:, ::-1]
            source_text = _decode_ids(x_test[example_id], id_to_char)
            target_text = _decode_ids(expected, id_to_char)
            prediction_text = _decode_ids(predicted, id_to_char)
            records.add_prediction(
                {
                    "epoch": 0,
                    "example_id": example_id,
                    "source": source_text,
                    "target": target_text,
                    "prediction": prediction_text,
                    "exact_match": int(predicted == expected),
                    "token_correct": sum(
                        left == right
                        for left, right in zip(predicted, expected, strict=True)
                    ),
                    "token_count": len(expected),
                }
            )
            render_examples.append(
                {
                    "example_id": example_id,
                    "source": source_text,
                    "target": target_text,
                    "prediction": prediction_text,
                    "source_labels": list(source_text),
                    "target_labels": list(target_text),
                }
            )
            for decode_step in range(weights.shape[0]):
                for encoder_position in range(weights.shape[1]):
                    records.add_attention(
                        {
                            "example_id": example_id,
                            "decode_step": decode_step,
                            "encoder_position": encoder_position,
                            "weight": float(weights[decode_step, encoder_position]),
                        }
                    )
        records.set_attention_render(
            {
                "source_checkpoint": str(Path(str(checkpoint_path))),
                "source_checkpoint_sha256": _path_digest(Path(str(checkpoint_path))),
                "example_selection_seed": int(
                    attention_config.get("selection_seed", 1984)
                ),
                "decode_policy": str(attention_config.get("decode", "greedy")),
                "conditioning": str(
                    attention_config.get(
                        "conditioning", attention_config.get("decode", "greedy")
                    )
                ),
                "condition": str(config.get("atomic_run_id", "")),
                "input_reversal": bool(dataset.get("reverse", True)),
                "x_axis": "encoder_position",
                "y_axis": "decode_step",
                "y_axis_inverted": True,
                "color_range": [0.0, 1.0],
                "examples": render_examples,
            }
        )
        records.flush()
        return ExperimentResult(
            metrics={
                "final/status/success": 1.0,
                "final/system/total_updates": 0.0,
                "final/system/completed_epochs": 0.0,
                "final/system/samples_seen": float(len(example_ids)),
            },
            artifact_root=artifact_root,
            model=model,
            metric_rows=(),
            profiling_metrics={},
        )


def _device_timer(config: dict[str, object], backend):
    profiling = _mapping(config, "profiling")
    return create_device_timer(
        backend, enabled=bool(profiling.get("device_timing", False))
    )


def _save_epoch_roles(manager: CheckpointManager) -> None:
    if manager.policy.save_latest:
        manager.save_latest()
    manager.save_periodic_if_due()


def _record_retained_checkpoints(
    records: DS2Records,
    manager: CheckpointManager,
    *,
    best_metric: str = "",
    best_value: float | None = None,
) -> None:
    final = manager.current("final")
    if final is not None:
        records.add_checkpoint(
            update=final.update,
            epoch=final.epoch,
            kind="final",
            path=final.path,
            sha256=final.sha256,
            checkpoint_id="final",
        )
    latest = manager.current("latest")
    if latest is not None:
        records.add_checkpoint(
            update=latest.update,
            epoch=latest.epoch,
            kind="latest",
            path=latest.path,
            sha256=latest.sha256,
            checkpoint_id=f"latest-epoch-{latest.epoch:04d}",
        )
    best = manager.current("best")
    if best is not None:
        records.add_checkpoint(
            update=best.update,
            epoch=best.epoch,
            kind="selected",
            path=best.path,
            sha256=best.sha256,
            checkpoint_id=f"selected-epoch-{best.epoch:04d}",
            selection_metric=best_metric,
            selection_value="" if best_value is None else best_value,
        )
    for periodic in manager.retained_periodic():
        records.add_checkpoint(
            update=periodic.update,
            epoch=periodic.epoch,
            kind="periodic",
            path=periodic.path,
            sha256=periodic.sha256,
            checkpoint_id=f"periodic-epoch-{periodic.epoch:04d}",
        )


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
                        questions[example_id : example_id + 1],
                        dtype=backend.xp.int64,
                    ),
                    backend=backend,
                )
                predicted = model.generate(question, start_id, len(expected))
            else:
                predicted = [int(value) for value in predictions[example_id]]
            records.add_prediction(
                {
                    "epoch": epoch,
                    "example_id": example_id,
                    "source": _decode_ids(questions[example_id], id_to_char),
                    "target": _decode_ids(expected, id_to_char),
                    "prediction": _decode_ids(predicted, id_to_char),
                    "exact_match": int(predicted == expected),
                    "token_correct": sum(
                        left == right
                        for left, right in zip(predicted, expected, strict=True)
                    ),
                    "token_count": len(expected),
                }
            )
    finally:
        model.train(was_training)


def _generate_attention_with_weights(
    model: AttentionSeq2seq, question: Tensor, start_id: int, sample_size: int, backend
) -> tuple[list[int], np.ndarray]:
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
            out = model.decoder.embed.forward(
                Tensor(sample_id.reshape((1, 1)), backend=backend)
            )
            dec_hs = model.decoder.lstm.forward(out)
            context = model.decoder.attention.forward(enc_hs, dec_hs)
            weights.append(model.decoder.attention.weights[0, 0].copy())
            score = model.decoder.affine.forward(
                Tensor(
                    xp.concatenate((context.data, dec_hs.data), axis=2), backend=backend
                )
            )
            sample_id = score.data.reshape(-1).argmax()
            sampled.append(sample_id)
        host_ids = (
            backend.to_numpy(xp.stack(sampled))
            if sampled
            else np.asarray([], dtype=np.int64)
        )
        host_weights = (
            backend.to_numpy(xp.stack(weights)) if weights else np.empty((0, 0))
        )
        return [int(value) for value in host_ids], np.asarray(host_weights)
    finally:
        model.train(was_training)


def _teacher_forced_attention_with_weights(
    model: AttentionSeq2seq,
    question: Tensor,
    target: np.ndarray,
    backend,
) -> tuple[list[int], np.ndarray]:
    """Run the book's teacher-forced decoder once and retain every score/weight."""
    was_training = bool(getattr(model, "training", True))
    model.train(False)
    try:
        decoder_x = Tensor(
            backend.xp.asarray(target[:-1][None, :], dtype=backend.xp.int64),
            backend=backend,
        )
        scores = model.forward(question, decoder_x, cache=True)
        weights = backend.to_numpy(model.decoder.attention.weights[0])
        predicted = backend.to_numpy(scores.data[0].argmax(axis=1))
        return [int(value) for value in predicted], np.asarray(weights)
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
    from repro_core.context.paths import RuntimePaths

    return RuntimePaths.from_environment().dataset("sequence") / file_name


def _array_digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


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
