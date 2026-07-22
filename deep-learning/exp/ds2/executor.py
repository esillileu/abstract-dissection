"""Portable PTB word2vec/RNNLM and character seq2seq experiment executors."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from mlprosection import Tensor
from mlprosection.datasets import load_ptb, load_sequence
from mlprosection.nn.model import Word2Vec
from mlprosection.nn.sampling import UnigramSampler
from mlprosection.nn.model.recurrent import AttentionSeq2seq, BetterRnnlm, PeekySeq2seq, Rnnlm, Seq2seq, VanillaRnnlm
from mlprosection.optim.SGD import Adam, SGD
from mlprosection.trainer import (
    LanguageModelTrainer,
    Seq2seqTrainer,
    Word2VecTrainer,
)

from mlprosection.experiment.contracts import ExperimentResult
from mlprosection.experiment.checkpoint import load_epoch_checkpoint, save_epoch_checkpoint
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


def _optimizer(config: dict[str, object], model):
    values = _mapping(config, "optimizer")
    name = str(values.get("name", "adam"))
    params = list(model.named_parameters())
    if name == "adam":
        return Adam(params, lr=float(values.get("learning_rate", 0.001)))
    if name == "sgd":
        return SGD(params, lr=float(values.get("learning_rate", 1.0)))
    raise ValueError(f"unsupported sequence optimizer: {name}")


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

    The accumulator stays on the active backend until a curve point is due, so
    recording one point does not force a host synchronization for every update.
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
    count = 0
    unit_count = 0
    update_start = None
    epoch_start = None

    def reduce(event):
        nonlocal total, count, unit_count, update_start, epoch_start
        if update_start is None:
            update_start = event.update
            epoch_start = event.epoch
        total = event.objective.data if total is None else total + event.objective.data
        count += 1
        unit_count += int(event.unit_count)
        if event.local_iteration % every != 0:
            return None
        value = event.objective.backend.scalar_to_float(total / count)
        if kind == "train_perplexity":
            value = float(event.objective.backend.xp.exp(value))
        point = {
            "series_id": kind,
            "plot_index": event.update - 1,
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
        total, count, unit_count = None, 0, 0
        update_start, epoch_start = None, None
        return point

    return reduce


@register_executor("word2vec")
class Word2VecExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset, model_config, training = (_mapping(config, key) for key in ("dataset", "model", "training"))
        corpus, word_to_id = _word2vec_corpus(_mapping(config, "dataset"))
        window = int(dataset.get("window_size", 5))
        contexts, targets = _contexts_targets(corpus, window)
        objective = str(model_config.get("objective", "negative_sampling"))
        sampler = None
        if objective == "negative_sampling":
            sampler_values = _mapping(model_config, "sampler")
            sampler = UnigramSampler.from_corpus(
                corpus,
                vocab_size=len(word_to_id),
                backend=backend,
                power=float(sampler_values.get("power", 0.75)),
                rejection_rounds=int(sampler_values.get("rejection_rounds", 4)),
            )
            context.metadata["negative_sampler"] = sampler.metadata
        model = Word2Vec(
            len(word_to_id), int(model_config.get("embedding_size", 100)),
            architecture=str(model_config.get("architecture", "cbow")),
            objective=objective,
            negative_samples=int(model_config.get("negative_samples", 5)),
            loss_reduction=str(_mapping(config, "loss").get("reduction", "mean")),
            sampler=sampler, backend=backend,
        )
        optimizer = _optimizer(config, model)
        seed_batch_order(backend, streams)
        batch_size, epochs = int(_mapping(config, "loader").get("batch_size", 100)), int(training.get("max_epochs", 10))
        x = backend.xp.asarray(contexts, dtype=backend.xp.int64)
        t = backend.xp.asarray(targets, dtype=backend.xp.int64)
        train_x, train_t = (t, x) if model.architecture == "skipgram" else (x, t)
        artifact_root = _artifact_root(config, context)
        records_sink = DS2Records()
        records_sink.bind_artifact_root(artifact_root)
        monitor = create_runtime_monitor(backend, _mapping(config, "profiling"))
        controller = EventExperimentExecutor(
            records=records_sink, evaluate=lambda _request: None,
            source_curve=_source_curve_from_objective(config),
            device_timer=_device_timer(config, backend),
        )
        trainer = Word2VecTrainer(
            model, optimizer, max_epochs=epochs, batch_size=batch_size,
            max_grad=_optional_max_grad(config), event_receivers=[controller],
        )
        with training_summary(monitor):
            records = controller.run(
                lambda: trainer.fit(Tensor(train_x, backend=backend), Tensor(train_t, backend=backend)),
                start_update=trainer.global_step + 1,
            )
        records.flush()
        final_loss = float(records.updates[-1]["loss"].backend.scalar_to_float(records.updates[-1]["loss"].data)) if records.updates else 0.0
        final_metrics = {"final/train/loss": final_loss}
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
        model = _language_model(str(model_config.get("alias")), len(ptb["word_to_id"]), model_config, backend)
        optimizer = _optimizer(config, model)
        seed_batch_order(backend, streams)
        train_corpus = ptb["train"][:int(dataset.get("train_limit", len(ptb["train"])))]
        train = Tensor(backend.xp.asarray(train_corpus[:-1], dtype=backend.xp.int64), backend=backend)
        train_targets = Tensor(backend.xp.asarray(train_corpus[1:], dtype=backend.xp.int64), backend=backend)
        max_epochs = int(training.get("max_epochs", 4))
        valid = Tensor(backend.xp.asarray(ptb["valid"][:-1], dtype=backend.xp.int64), backend=backend)
        valid_targets = Tensor(backend.xp.asarray(ptb["valid"][1:], dtype=backend.xp.int64), backend=backend)
        test = Tensor(backend.xp.asarray(ptb["test"][:-1], dtype=backend.xp.int64), backend=backend)
        test_targets = Tensor(backend.xp.asarray(ptb["test"][1:], dtype=backend.xp.int64), backend=backend)
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
            model, optimizer, max_epochs=max_epochs,
            batch_size=int(loader.get("batch_size", 20)), time_size=int(loader.get("time_size", 35)),
            max_grad=_optional_max_grad(config, default=0.25),
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
                        selected_checkpoint_path = save_epoch_checkpoint(root=Path(str(context.metadata["checkpoint_root"])), model=model, optimizer=optimizer, trainer=trainer, config_digest=config_digest)
                        records_sink.add_checkpoint(
                            update=trainer.global_step, epoch=step, kind="selected",
                            path=selected_checkpoint_path, sha256=_path_digest(selected_checkpoint_path),
                            checkpoint_id=f"selected-epoch-{step:04d}",
                            selection_metric="valid/perplexity",
                            selection_value=valid_ppl,
                        )
                        records_sink.flush()
                elif str(model_config.get("alias")) == "BetterRnnlm" and str(_mapping(config, "scheduler").get("name", "constant")) == "validation_decay":
                    optimizer.lr /= float(_mapping(config, "scheduler").get("factor", 4.0))
            else:
                test_ppl = float(result.perplexity)
        controller = EventExperimentExecutor(
            records=records_sink, evaluate=evaluate_request,
            epoch_requests=epoch_requests, after_evaluation=after_evaluation,
            source_curve=_source_curve_from_objective(config),
            device_timer=_device_timer(config, backend),
        )
        with training_summary(monitor):
            records = controller.run(lambda: trainer.fit(train, train_targets))
        records.flush()
        if test_at_end or test_ppl == float("inf"):
            if bool(_mapping(config, "checkpoint").get("save_best", False)):
                if selected_checkpoint_path is None:
                    raise RuntimeError("selected checkpoint is required before terminal test evaluation")
                load_epoch_checkpoint(path=selected_checkpoint_path, model=model, optimizer=optimizer, trainer=trainer, config_digest=config_digest)
            test_ppl = float(trainer.evaluate(test, test_targets).perplexity)
        final_train_ppl = float(backend.xp.exp(records.updates[-1]["loss"].backend.scalar_to_float(records.updates[-1]["loss"].data))) if records.updates else float("inf")
        final_metrics = {
            "final/train/perplexity": final_train_ppl,
            "final/test/perplexity": test_ppl,
            "final/train/ppl": final_train_ppl,
            "final/test/ppl": test_ppl,
        }
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
        context.metadata["data"] = {
            "split_seed": split_seed,
            "split_algorithm": split_algorithm,
        }
        x_train, t_train = data["train"]
        x_test, t_test = data["test"]
        if bool(dataset.get("reverse", False)):
            x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
        model = _seq_model(str(model_config.get("alias")), len(data["char_to_id"]), model_config, backend)
        optimizer = _optimizer(config, model)
        seed_batch_order(backend, streams)
        batch_size, epochs = int(loader.get("batch_size", 128)), int(training.get("max_epochs", 10))
        train_x = Tensor(backend.xp.asarray(x_train, dtype=backend.xp.int64), backend=backend)
        train_t = Tensor(backend.xp.asarray(t_train, dtype=backend.xp.int64), backend=backend)
        test_source = (Tensor(backend.xp.asarray(x_test, dtype=backend.xp.int64), backend=backend), Tensor(backend.xp.asarray(t_test, dtype=backend.xp.int64), backend=backend))
        request = EvaluationRequest("sequence-test-full", "test", test_source, ("exact_match_accuracy", "token_accuracy"))
        trainer = Seq2seqTrainer(
            model, optimizer, max_epochs=epochs, batch_size=batch_size,
            start_id=data["char_to_id"]["_"], max_grad=_optional_max_grad(config, default=5.0),
        )
        artifact_root = _artifact_root(config, context)
        records = DS2Records()
        records.bind_artifact_root(artifact_root)
        monitor = create_runtime_monitor(backend, _mapping(config, "profiling"))
        checkpoint_config = _mapping(config, "checkpoint")
        config_digest = _config_digest(config)
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
                if bool(checkpoint_config.get("save_best", False)) and result.exact_match_accuracy > best_exact:
                    best_exact = float(result.exact_match_accuracy)
                    records.flush()
                    path = save_epoch_checkpoint(root=Path(str(context.metadata["checkpoint_root"])), model=model, optimizer=optimizer, trainer=trainer, config_digest=config_digest)
                    records.add_checkpoint(
                        update=trainer.global_step, epoch=step, kind="selected",
                        path=path, sha256=_path_digest(path),
                        checkpoint_id=f"selected-epoch-{step:04d}",
                        selection_metric="test/exact_match_accuracy",
                        selection_value=best_exact,
                    )
                    records.flush()

        controller = EventExperimentExecutor(
            records=records,
            evaluate=lambda _request: trainer.evaluate(*test_source, metrics=("exact_match_accuracy", "token_accuracy")),
            epoch_requests=lambda _event: (request,),
            after_evaluation=after_evaluation,
            device_timer=_device_timer(config, backend),
        )
        with training_summary(monitor):
            records = controller.run(lambda: trainer.fit(train_x, train_t))
        _record_seq_predictions(records, model, x_test, t_test, data["char_to_id"], data["id_to_char"], backend, _mapping(config, "recording"))
        records.flush()
        last_evaluation = records.evaluations[-2:] if len(records.evaluations) >= 2 else []
        final_values = {"final/train/loss": float(records.updates[-1]["loss"].backend.scalar_to_float(records.updates[-1]["loss"].data)) if records.updates else 0.0}
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
        data = load_sequence(
            str(dataset["file"]),
            seed=int(dataset.get("split_seed", 1984)),
            split_algorithm=str(dataset.get("split_algorithm", "legacy_numpy_randomstate")),
        )
        x_test, t_test = data["test"]
        if bool(dataset.get("reverse", True)):
            x_test = x_test[:, ::-1]
        model = _seq_model(str(model_config.get("alias", "AttentionSeq2seq")), len(data["char_to_id"]), model_config, backend)
        if not isinstance(model, AttentionSeq2seq):
            raise ValueError("DS2 GO01 requires AttentionSeq2seq model")
        _load_model_checkpoint(model, Path(str(checkpoint_path)))
        artifact_root = _artifact_root(config, context)
        records = DS2Records()
        records.bind_artifact_root(artifact_root)
        recording = _mapping(config, "recording")
        attention_config = _mapping(recording, "attention")
        count = int(attention_config.get("count", 5))
        example_ids = list(range(min(count, len(x_test))))
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
                "example_id": example_id,
                "source": source_text,
                "target": target_text,
                "prediction": prediction_text,
                "exact_match": int(predicted == expected),
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


def _contexts_targets(corpus, window: int):
    import numpy as np
    centers = np.arange(window, len(corpus) - window)
    contexts = np.stack([np.concatenate((corpus[index - window:index], corpus[index + 1:index + window + 1])) for index in centers])
    return contexts, corpus[window:-window]


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


def _word2vec_prediction_term_count(model: Word2Vec, window: int) -> int:
    """Return the sigmoid terms summed by the book's negative-sampling loss."""
    contexts = 2 * window if model.architecture == "skipgram" else 1
    return contexts * (model.negative_samples + 1)


def _record_seq_predictions(records: DS2Records, model, questions, answers, char_to_id, id_to_char, backend, recording: dict[str, object]) -> None:
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
            question = Tensor(backend.xp.asarray(questions[example_id:example_id + 1], dtype=backend.xp.int64), backend=backend)
            expected = [int(value) for value in answers[example_id][1:]]
            predicted = model.generate(question, start_id, len(expected))
            records.add_prediction({
                "example_id": example_id,
                "source": _decode_ids(questions[example_id], id_to_char),
                "target": _decode_ids(expected, id_to_char),
                "prediction": _decode_ids(predicted, id_to_char),
                "exact_match": int(predicted == expected),
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
        sample_id = start_id
        sampled = []
        weights = []
        for _ in range(sample_size):
            out = model.decoder.embed.forward(Tensor(xp.asarray([[sample_id]], dtype=xp.int64), backend=backend))
            dec_hs = model.decoder.lstm.forward(out)
            context = model.decoder.attention.forward(enc_hs, dec_hs)
            weights.append(backend.to_numpy(model.decoder.attention.weights[0, 0]).copy())
            score = model.decoder.affine.forward(Tensor(xp.concatenate((context.data, dec_hs.data), axis=2), backend=backend))
            sample_id = int(score.data.reshape(-1).argmax())
            sampled.append(sample_id)
        return sampled, np.asarray(weights)
    finally:
        model.train(was_training)


def _decode_ids(values, id_to_char: dict[int, str]) -> str:
    return "".join(id_to_char[int(value)] for value in values)


def _load_model_checkpoint(model, path: Path) -> None:
    if path.is_dir():
        model.load_params_npz(path / "model.npz")
    elif path.is_file():
        model.load_params_npz(path)
    else:
        raise FileNotFoundError(path)


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


def _save_evaluation_checkpoint(config: dict[str, object], context: ExperimentContext, *, model, optimizer, trainer) -> None:
    if not bool(_mapping(config, "checkpoint").get("save_on_eval", False)):
        return
    checkpoint_config = dict(_mapping(config, "checkpoint"))
    checkpoint_config.pop("resume", None)
    identity = dict(config)
    identity["checkpoint"] = checkpoint_config
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    root = Path(str(context.metadata["checkpoint_root"]))
    path = save_epoch_checkpoint(root=root, model=model, optimizer=optimizer, trainer=trainer, config_digest=digest)
    callback = context.metadata.get("record_eval_checkpoint")
    if callable(callback):
        callback(path)


def _language_model(alias: str, vocab_size: int, values: dict[str, object], backend):
    kwargs = {"vocab_size": vocab_size, "wordvec_size": int(values.get("wordvec_size", 100)), "hidden_size": int(values.get("hidden_size", 100)), "backend": backend}
    if alias == "VanillaRnnlm":
        return VanillaRnnlm(**kwargs)
    if alias == "Rnnlm":
        return Rnnlm(**kwargs)
    if alias == "BetterRnnlm":
        return BetterRnnlm(**kwargs, dropout_ratio=float(values.get("dropout_ratio", 0.5)))
    raise ValueError(f"unknown language-model alias: {alias}")


def _seq_model(alias: str, vocab_size: int, values: dict[str, object], backend):
    kwargs = {"vocab_size": vocab_size, "wordvec_size": int(values.get("wordvec_size", 16)), "hidden_size": int(values.get("hidden_size", 128)), "backend": backend}
    if alias == "Seq2seq":
        return Seq2seq(**kwargs)
    if alias == "PeekySeq2seq":
        return PeekySeq2seq(**kwargs)
    if alias == "AttentionSeq2seq":
        return AttentionSeq2seq(**kwargs)
    raise ValueError(f"unknown seq2seq alias: {alias}")


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
