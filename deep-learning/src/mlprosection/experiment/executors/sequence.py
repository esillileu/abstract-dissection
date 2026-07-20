"""Portable PTB word2vec/RNNLM and character seq2seq experiment executors."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from mlprosection import Tensor
from mlprosection.datasets import load_ptb, load_sequence
from mlprosection.nn.model import Word2Vec
from mlprosection.nn.model.recurrent import AttentionSeq2seq, BetterRnnlm, PeekySeq2seq, Rnnlm, Seq2seq, VanillaRnnlm
from mlprosection.optim.SGD import Adam, SGD
from mlprosection.optim.transform import ClipGradNorm

from ..contracts import ExperimentResult
from ..executor import ExperimentContext
from ..registry import register_executor
from ..reproducibility import configure_runtime, seed_batch_order


def _mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _artifact_root(config: dict[str, object]) -> Path:
    return Path(str(_mapping(config, "run").get("artifact_root", "experiments/results/runs"))) / str(config["atomic_run_id"])


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


@register_executor("word2vec")
class Word2VecExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset, model_config, training = (_mapping(config, key) for key in ("dataset", "model", "training"))
        ptb = load_ptb()
        corpus = ptb["train"]
        window = int(dataset.get("window_size", 5))
        contexts, targets = _contexts_targets(corpus, window)
        counts = backend.xp.asarray([int((corpus == index).sum()) for index in range(len(ptb["word_to_id"]))], dtype=backend.float_dtype)
        distribution = counts ** 0.75
        model = Word2Vec(
            len(ptb["word_to_id"]), int(model_config.get("embedding_size", 100)),
            architecture=str(model_config.get("architecture", "cbow")),
            objective=str(model_config.get("objective", "negative_sampling")),
            negative_samples=int(model_config.get("negative_samples", 5)),
            sampling_distribution=distribution, backend=backend,
        )
        optimizer = _optimizer(config, model)
        seed_batch_order(backend, streams)
        batch_size, epochs, interval = int(_mapping(config, "loader").get("batch_size", 100)), int(training.get("max_epochs", 10)), int(training.get("log_interval", 1000))
        x = backend.xp.asarray(contexts, dtype=backend.xp.int64)
        t = backend.xp.asarray(targets, dtype=backend.xp.int64)
        history: list[tuple[str, int, str, float]] = []
        step = 0
        for epoch in range(epochs):
            order = backend.xp.random.permutation(len(x))
            total = 0.0
            for start in range(0, len(x) - batch_size + 1, batch_size):
                indices = order[start:start + batch_size]
                if model.architecture == "skipgram":
                    batch_x = Tensor(t[indices], backend=backend)
                    batch_t = Tensor(x[indices], backend=backend)
                else:
                    batch_x = Tensor(x[indices], backend=backend)
                    batch_t = Tensor(t[indices], backend=backend)
                loss = model.forward(batch_x, batch_t)
                model.backward()
                optimizer.update()
                step += 1
                total += float(loss.data)
                if step % interval == 0:
                    value = total / interval
                    history.append(("step", step, "train/normalized_loss", value))
                    context.emit_metric(step, {"train/normalized_loss": value})
                    total = 0.0
            if total:
                history.append(("epoch", epoch + 1, "train/normalized_loss", total / max(1, step % interval)))
        final_loss = history[-1][3] if history else 0.0
        return ExperimentResult(
            metrics=_final(updates=step, epochs=epochs, samples=len(x) * epochs, **{"final/train/normalized_loss": final_loss}),
            artifact_root=_artifact_root(config), model=model, history=tuple(history),
        )


@register_executor("language_modeling")
class LanguageModelExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        from mlprosection.trainer.time_trainer import TimeTrainer

        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        ptb = load_ptb()
        model_config, loader, training = (_mapping(config, key) for key in ("model", "loader", "training"))
        model = _language_model(str(model_config.get("alias")), len(ptb["word_to_id"]), model_config, backend)
        optimizer = _optimizer(config, model)
        seed_batch_order(backend, streams)
        train = Tensor(backend.xp.asarray(ptb["train"][:-1], dtype=backend.xp.int64), backend=backend)
        train_targets = Tensor(backend.xp.asarray(ptb["train"][1:], dtype=backend.xp.int64), backend=backend)
        max_epochs = int(training.get("max_epochs", 4))
        trainer = TimeTrainer(model, optimizer, max_epoch=max_epochs, batch_size=int(loader.get("batch_size", 20)), time_size=int(loader.get("time_size", 35)), log_interval=int(training.get("log_interval", 20)), max_grad=float(_mapping(config, "policy").get("max_grad", 0.25)), callbacks=[_Callback(context)])
        valid = Tensor(backend.xp.asarray(ptb["valid"][:-1], dtype=backend.xp.int64), backend=backend)
        valid_targets = Tensor(backend.xp.asarray(ptb["valid"][1:], dtype=backend.xp.int64), backend=backend)
        test = Tensor(backend.xp.asarray(ptb["test"][:-1], dtype=backend.xp.int64), backend=backend)
        test_targets = Tensor(backend.xp.asarray(ptb["test"][1:], dtype=backend.xp.int64), backend=backend)
        valid_ppl = float("inf")
        test_ppl = float("inf")
        best_valid = float("inf")
        validation_history: list[tuple[str, int, str, float]] = []
        checkpoint_root = Path(str(context.metadata.get("checkpoint_root", "experiments/results/checkpoints")))
        for target_epoch in range(1, max_epochs + 1):
            trainer.max_epoch = target_epoch
            trainer.fit(train, train_targets)
            valid_ppl = trainer.evaluate_perplexity(valid, valid_targets)
            test_ppl = trainer.evaluate_perplexity(test, test_targets)
            validation_history.extend((("epoch", target_epoch, "valid/perplexity", valid_ppl), ("epoch", target_epoch, "test/perplexity", test_ppl)))
            if valid_ppl < best_valid:
                best_valid = valid_ppl
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                model.save_params_npz(checkpoint_root / "best.npz")
            elif str(model_config.get("alias")) == "BetterRnnlm":
                optimizer.lr /= float(_mapping(config, "scheduler").get("factor", 4.0))
        trainer.max_epoch = max_epochs
        history = [("step", index + 1, "train/perplexity", value) for index, value in enumerate(trainer.history.train_ppl)]
        history += validation_history
        return ExperimentResult(metrics=_final(updates=trainer.global_step, epochs=trainer.epoch, samples=len(train) * trainer.epoch, **{"final/train/perplexity": trainer.history.train_ppl[-1], "final/valid/perplexity": valid_ppl, "final/test/perplexity": test_ppl}), artifact_root=_artifact_root(config), model=model, history=tuple(history))


@register_executor("seq2seq")
class Seq2SeqExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset, model_config, loader, training = (_mapping(config, key) for key in ("dataset", "model", "loader", "training"))
        data = load_sequence(str(dataset["file"]), seed=streams.dataset_split)
        x_train, t_train = data["train"]
        x_test, t_test = data["test"]
        if bool(dataset.get("reverse", False)):
            x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
        model = _seq_model(str(model_config.get("alias")), len(data["char_to_id"]), model_config, backend)
        optimizer = _optimizer(config, model)
        seed_batch_order(backend, streams)
        batch_size, epochs = int(loader.get("batch_size", 128)), int(training.get("max_epochs", 10))
        history: list[tuple[str, int, str, float]] = []
        for epoch in range(1, epochs + 1):
            order = backend.xp.random.permutation(len(x_train))
            total = 0.0
            count = 0
            model.train(True)
            for start in range(0, len(x_train) - batch_size + 1, batch_size):
                indices = order[start:start + batch_size]
                loss = model.forward(Tensor(backend.xp.asarray(x_train[indices], dtype=backend.xp.int64), backend=backend), Tensor(backend.xp.asarray(t_train[indices], dtype=backend.xp.int64), backend=backend))
                model.backward()
                ClipGradNorm(float(_mapping(config, "policy").get("max_grad", 5.0)))(list(model.named_parameters()))
                optimizer.update()
                total += float(loss.data)
                count += 1
            exact, token = _seq_accuracy(model, x_test, t_test, data["char_to_id"], backend)
            train_loss = total / max(count, 1)
            history.extend((("epoch", epoch, "train/loss", train_loss), ("epoch", epoch, "test/exact_match", exact), ("epoch", epoch, "test/token_accuracy", token)))
            context.emit_metric(epoch, {"train/loss": train_loss, "test/exact_match": exact, "test/token_accuracy": token})
        attention_entropy = _save_attention_artifact(model, x_test, t_test, backend, context)
        final_values = {"final/train/loss": history[-3][3], "final/test/exact_match": history[-2][3], "final/test/token_accuracy": history[-1][3]}
        if attention_entropy is not None:
            final_values["final/attention/entropy"] = attention_entropy
        return ExperimentResult(metrics=_final(updates=count * epochs, epochs=epochs, samples=len(x_train) * epochs, **final_values), artifact_root=_artifact_root(config), model=model, history=tuple(history))


class _Callback:
    def __init__(self, context: ExperimentContext) -> None:
        self.context = context
    def on_batch_end(self, *, step: int) -> None:
        pass
    def on_interval(self, *, metrics: dict[str, float]) -> None:
        self.context.emit_metric(int(metrics["iteration"]), metrics)
    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None:
        self.context.emit_metric(epoch, metrics)


def _contexts_targets(corpus, window: int):
    import numpy as np
    centers = np.arange(window, len(corpus) - window)
    contexts = np.stack([np.concatenate((corpus[index - window:index], corpus[index + 1:index + window + 1])) for index in centers])
    return contexts, corpus[window:-window]


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
    root = Path(str(context.metadata.get("checkpoint_root", "experiments/results/checkpoints"))).parent
    path = root / "analysis" / "attention_map.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, attention=values, entropy=entropy)
    model.train(True)
    return entropy
