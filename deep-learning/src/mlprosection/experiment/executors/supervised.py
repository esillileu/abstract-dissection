from __future__ import annotations

# ruff: noqa: E701, E702

from pathlib import Path
from dataclasses import asdict
import hashlib
import json

from mlprosection.datasets import load_mnist
from mlprosection.datasets.mnist import save_file as mnist_cache_path
from mlprosection.nn.layers import BatchNormalization, SoftmaxWithLoss
from mlprosection.nn.model import DeepCNN, MLP, SimpleCNN
from mlprosection.optim.SGD import AdaGrad, Adam, Momentum, SGD
from mlprosection.optim.transform import L2Regularization
from mlprosection.trainer import ForwardTrainer

from ..contracts import ExperimentResult
from ..checkpoint import load_epoch_checkpoint, save_epoch_checkpoint
from ..executor import ExperimentContext
from ..metrics import build_final_metrics, epoch_history, evaluation_history, update_history
from ..registry import register_executor
from ..reproducibility import configure_runtime, seed_batch_order


@register_executor("supervised_classification")
class SupervisedClassificationExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        dataset, model_config, training_config, loader_config, optimizer_config = (_mapping(config, key) for key in ("dataset", "model", "training", "loader", "optimizer"))
        backend, streams, actual_runtime = configure_runtime(config)
        context.metadata["runtime"] = actual_runtime
        context.metadata["seed_streams"] = asdict(streams)
        flatten = bool(dataset.get("flatten", True)); gpu = backend.is_gpu
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=flatten, gpu=gpu)
        if (limit := dataset.get("train_limit")) is not None: x_train, t_train = x_train[:int(limit)], t_train[:int(limit)]
        if (limit := dataset.get("test_limit")) is not None: x_test, t_test = x_test[:int(limit)], t_test[:int(limit)]
        context.metadata["data"] = {
            "cache_path": mnist_cache_path,
            "cache_sha256": _file_digest(Path(mnist_cache_path)),
            "train_samples": len(x_train), "test_samples": len(x_test), "flatten": flatten,
        }
        model = _model(model_config)
        if any(isinstance(layer, BatchNormalization) for layer in model.children()):
            model.forward(x_train[:1])
        optimizer = _optimizer(str(optimizer_config.get("name", "sgd")), list(model.named_parameters()), float(optimizer_config.get("learning_rate", 0.01)), float(optimizer_config.get("weight_decay", 0.0)))
        checkpoint_config = _mapping(config, "checkpoint")
        checkpoint_root = Path(str(context.metadata.get("checkpoint_root", "experiments/results/checkpoints")))
        checkpoint_identity = dict(config)
        checkpoint_identity["checkpoint"] = dict(checkpoint_config)
        checkpoint_identity["checkpoint"].pop("resume", None)
        config_digest = hashlib.sha256(json.dumps(checkpoint_identity, sort_keys=True, default=str).encode()).hexdigest()
        trainer = ForwardTrainer(model, SoftmaxWithLoss().to(model.backend), optimizer, max_epoch=int(training_config.get("max_epochs", 1)), batch_size=int(loader_config.get("batch_size", 32)), log_interval=int(training_config.get("log_interval", 20)), callbacks=[_Callback(context)], on_epoch_checkpoint=lambda: save_epoch_checkpoint(root=checkpoint_root, model=model, optimizer=optimizer, trainer=trainer, config_digest=config_digest))
        if (resume := checkpoint_config.get("resume")):
            load_epoch_checkpoint(path=str(resume), model=model, optimizer=optimizer, trainer=trainer, config_digest=config_digest)
        seed_batch_order(backend, streams)
        trainer.fit(x_train, t_train, x_test, t_test)
        profiling = trainer.profiling_metrics()
        metrics = build_final_metrics(
            train_loss=float(trainer.losses.train[-1]), test_loss=float(trainer.losses.valid[-1]),
            train_accuracy=float(trainer.accuracies.train[-1]), test_accuracy=float(trainer.accuracies.valid[-1]),
            profiling_metrics=profiling, total_updates=trainer.global_step,
            completed_epochs=int(training_config.get("max_epochs", 1)), samples_seen=len(x_train) * int(training_config.get("max_epochs", 1)),
        )
        history = update_history(train_logs=trainer.logs.train)
        history += evaluation_history(valid_logs=trainer.logs.valid)
        history += epoch_history(
            train_losses=trainer.losses.train, test_losses=trainer.losses.valid,
            train_accuracies=trainer.accuracies.train, test_accuracies=trainer.accuracies.valid,
        )
        return ExperimentResult(metrics=metrics, artifact_root=_artifact_root(config), model=model, history=tuple(history), profiling_metrics=profiling)


class _Callback:
    def __init__(self, context: ExperimentContext) -> None: self.context = context
    def on_batch_end(self, *, step: int) -> None: pass
    def on_interval(self, *, metrics: dict[str, float]) -> None: self.context.emit_metric(int(metrics["iteration"]), metrics)
    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None: self.context.emit_metric(epoch, metrics)


def _model(config: dict[str, object]):
    name = str(config.get("alias", config.get("name", "MLP")))
    values = {key: value for key, value in config.items() if key not in {"alias", "name", "family", "task_type", "input_shape", "output_shape", "structure_signature", "use_batchnorm", "use_dropout", "num_hidden_layers", "num_conv_layers", "normalization", "model/flops", "model/macs"}}
    if name == "MLP":
        if "activation" in values:
            values["activation_name"] = values.pop("activation")
        if "use_batchnorm" in config:
            values["batchnorm"] = bool(config["use_batchnorm"])
    else:
        values.pop("activation", None)
    if name == "MLP": return MLP(**values)
    if name == "SimpleCNN": return SimpleCNN(**values)
    if name == "DeepCNN": return DeepCNN(**values)
    raise ValueError(f"unknown model alias: {name}")


def _optimizer(name: str, params, learning_rate: float, weight_decay: float):
    hooks = [L2Regularization(weight_decay)] if weight_decay else None
    if name == "sgd": return SGD(params, lr=learning_rate, pre_step_hooks=hooks)
    if name == "momentum": return Momentum(params, lr=learning_rate, momentum=0.9, pre_step_hooks=hooks)
    if name == "adagrad": return AdaGrad(params, lr=learning_rate, pre_step_hooks=hooks)
    if name == "adam": return Adam(params, lr=learning_rate, pre_step_hooks=hooks)
    raise ValueError(f"unknown optimizer: {name}")


def _mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if not isinstance(value, dict): raise ValueError(f"{key} must be a mapping")
    return value


def _artifact_root(config: dict[str, object]) -> Path:
    run = _mapping(config, "run")
    return Path(str(run.get("artifact_root", "experiments/results/runs"))) / str(run.get("name", config["kind"]))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
