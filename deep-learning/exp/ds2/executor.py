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
from mlprosection.experiment.checkpoint import save_epoch_checkpoint
from mlprosection.experiment.event_executor import EvaluationRequest, EventExperimentExecutor
from mlprosection.experiment.executor import ExperimentContext
from mlprosection.experiment.registry import register_executor
from mlprosection.experiment.reproducibility import configure_runtime, seed_batch_order

from .records import DS2Records


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
    total = None
    count = 0

    def reduce(event):
        nonlocal total, count
        total = event.objective.data if total is None else total + event.objective.data
        count += 1
        if event.local_iteration % every != 0:
            return None
        value = event.objective.backend.scalar_to_float(total / count)
        total, count = None, 0
        if kind == "train_perplexity":
            value = float(event.objective.backend.xp.exp(value))
        return {
            "series_id": kind,
            "plot_index": event.update - 1,
            "update_end": event.update,
            "epoch": event.epoch,
            "value": value,
        }

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
        controller = EventExperimentExecutor(
            records=DS2Records(), evaluate=lambda _request: None,
            source_curve=_source_curve_from_objective(config),
        )
        trainer = Word2VecTrainer(
            model, optimizer, max_epochs=epochs, batch_size=batch_size,
            max_grad=_optional_max_grad(config), event_receivers=[controller],
        )
        records = controller.run(
            lambda: trainer.fit(Tensor(train_x, backend=backend), Tensor(train_t, backend=backend)),
            start_update=trainer.global_step + 1,
        )
        artifact_root = _artifact_root(config, context)
        records.write_csv(artifact_root)
        history = list(records.history_rows())
        final_loss = float(records.updates[-1]["loss"].backend.scalar_to_float(records.updates[-1]["loss"].data)) if records.updates else 0.0
        final_metrics = {"final/train/loss": final_loss}
        return ExperimentResult(
            metrics=_final(updates=trainer.global_step, epochs=trainer.epoch, samples=len(x) * trainer.epoch, **final_metrics),
            artifact_root=artifact_root, model=model, history=tuple(history),
            profiling_metrics={},
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
        def after_evaluation(request, result, _axis, step):
            nonlocal valid_ppl, test_ppl, best_valid, best_valid_epoch
            if request.split == "valid":
                valid_ppl = float(result.perplexity)
                if valid_ppl < best_valid:
                    best_valid, best_valid_epoch = valid_ppl, step
                elif str(model_config.get("alias")) == "BetterRnnlm" and str(_mapping(config, "scheduler").get("name", "constant")) == "validation_decay":
                    optimizer.lr /= float(_mapping(config, "scheduler").get("factor", 4.0))
            else:
                test_ppl = float(result.perplexity)
        controller = EventExperimentExecutor(
            records=DS2Records(), evaluate=evaluate_request,
            epoch_requests=epoch_requests, after_evaluation=after_evaluation,
            source_curve=_source_curve_from_objective(config),
        )
        records = controller.run(lambda: trainer.fit(train, train_targets))
        artifact_root = _artifact_root(config, context)
        records.write_csv(artifact_root)
        if test_at_end or test_ppl == float("inf"):
            test_ppl = float(trainer.evaluate(test, test_targets).perplexity)
        history = list(records.history_rows())
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
        return ExperimentResult(metrics=_final(updates=trainer.global_step, epochs=trainer.epoch, samples=len(train) * trainer.epoch, **final_metrics), artifact_root=artifact_root, model=model, history=tuple(history), profiling_metrics={})


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
        records = DS2Records()

        def after_evaluation(_request, result, axis, step):
            if axis == "epoch" and result.exact_match_accuracy is not None:
                records.add_source_curve({
                    "series_id": "full_test_exact_match",
                    "plot_index": step - 1,
                    "update_end": trainer.global_step,
                    "epoch": step,
                    "value": result.exact_match_accuracy,
                })

        controller = EventExperimentExecutor(
            records=records,
            evaluate=lambda _request: trainer.evaluate(*test_source, metrics=("exact_match_accuracy", "token_accuracy")),
            epoch_requests=lambda _event: (request,),
            after_evaluation=after_evaluation,
        )
        records = controller.run(lambda: trainer.fit(train_x, train_t))
        artifact_root = _artifact_root(config, context)
        records.write_csv(artifact_root)
        history = list(records.history_rows())
        attention_entropy = _save_attention_artifact(model, x_test, t_test, backend, context)
        last_evaluation = records.evaluations[-2:] if len(records.evaluations) >= 2 else []
        final_values = {"final/train/loss": float(records.updates[-1]["loss"].backend.scalar_to_float(records.updates[-1]["loss"].data)) if records.updates else 0.0}
        for row in last_evaluation:
            final_values[f"final/test/{row['metric'].replace('_accuracy', '')}"] = float(row["value"])
        if attention_entropy is not None:
            final_values["final/attention/entropy"] = attention_entropy
            final_values["final/attention/entropy_mean"] = attention_entropy
        return ExperimentResult(metrics=_final(updates=trainer.global_step, epochs=trainer.epoch, samples=len(x_train) * trainer.epoch, **final_values), artifact_root=artifact_root, model=model, history=tuple(history), profiling_metrics={})


class _Callback:
    def __init__(self, context: ExperimentContext, *, loss_metric: str | None = None) -> None:
        self.context = context
        self.loss_metric = loss_metric
    def on_batch_end(self, *, step: int) -> None:
        pass
    def on_interval(self, *, metrics: dict[str, float]) -> None:
        self.context.emit_metric(int(metrics.get("global_step", metrics["iteration"])), self._map_loss(metrics))
    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None:
        self.context.emit_metric(epoch, self._map_loss(metrics))

    def _map_loss(self, metrics: dict[str, float]) -> dict[str, float]:
        if self.loss_metric is None:
            return metrics
        mapped = dict(metrics)
        if "loss" in mapped:
            mapped[self.loss_metric] = mapped.pop("loss")
        if "train/loss" in mapped:
            mapped[self.loss_metric] = mapped.pop("train/loss")
        if "normalized_loss" in mapped:
            mapped["train/normalized_loss"] = mapped.pop("normalized_loss")
        return mapped


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
    root = Path(str(context.metadata.get("artifact_root", "experiments/results/runs")))
    path = root / "analysis" / "attention_map.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, attention=values, entropy=entropy)
    model.train(True)
    return entropy
