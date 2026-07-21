from __future__ import annotations

# ruff: noqa: E701, E702

from pathlib import Path
from dataclasses import asdict
import hashlib
import json

import numpy as np

from mlprosection.datasets import load_mnist
from mlprosection.datasets.mnist import save_file as mnist_cache_path
from mlprosection.nn.layers import BatchNormalization, SoftmaxWithLoss
from mlprosection.nn.model import DeepCNN, MLP, SimpleCNN
from mlprosection.optim.SGD import AdaGrad, Adam, Momentum, SGD
from mlprosection.optim.transform import L2Regularization
from mlprosection.trainer import ForwardTrainer
from mlprosection.profiling import profiling_config_from_mapping

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
        transform_metadata = _apply_input_transform(
            dataset=dataset,
            backend=backend,
            x_train=x_train,
            x_test=x_test,
            artifact_root=Path(str(context.metadata["artifact_root"])),
        )
        if transform_metadata["name"] == "pixel_permutation":
            permutation = transform_metadata.pop("permutation")
            x_train, x_test = _permute_pixels(x_train, permutation), _permute_pixels(x_test, permutation)
        x_train, t_train, x_valid, t_valid, validation_metadata = _validation_probe(
            dataset=dataset, backend=backend, x_train=x_train, t_train=t_train,
            artifact_root=Path(str(context.metadata["artifact_root"])),
        )
        context.metadata["data"] = {
            "cache_path": mnist_cache_path,
            "cache_sha256": _file_digest(Path(mnist_cache_path)),
            "train_samples": len(x_train), "test_samples": len(x_test), "flatten": flatten,
            "input_transform": transform_metadata, "validation": validation_metadata,
        }
        model = _model(model_config)
        if any(isinstance(layer, BatchNormalization) for layer in model.children()):
            model.forward(x_train[:1])
        optimizer = _optimizer(optimizer_config, list(model.named_parameters()))
        checkpoint_config = _mapping(config, "checkpoint")
        checkpoint_identity = dict(config)
        checkpoint_identity["checkpoint"] = dict(checkpoint_config)
        checkpoint_identity["checkpoint"].pop("resume", None)
        config_digest = hashlib.sha256(json.dumps(checkpoint_identity, sort_keys=True, default=str).encode()).hexdigest()
        max_updates = training_config.get("max_updates")
        epoch_test_metrics: list[dict[str, float]] = []
        def save_evaluation_checkpoint() -> None:
            # Full official-test evaluation is intentionally epoch-bound.  The
            # frequent curve uses only the fixed validation probe.
            epoch_test_metrics.append(trainer._evaluate_split(x_test, t_test))
            if not bool(checkpoint_config.get("save_on_eval", False)):
                return
            path = save_epoch_checkpoint(
                root=Path(str(context.metadata["checkpoint_root"])),
                model=model,
                optimizer=optimizer,
                trainer=trainer,
                config_digest=config_digest,
            )
            callback = context.metadata.get("record_eval_checkpoint")
            if callable(callback):
                callback(path)

        trainer = ForwardTrainer(model, SoftmaxWithLoss().to(model.backend), optimizer, max_epoch=int(training_config.get("max_epochs", 1)), max_updates=None if max_updates is None else int(max_updates), batch_size=int(loader_config.get("batch_size", 32)), log_interval=int(training_config.get("log_interval", 20)), drop_last=bool(loader_config.get("drop_last", False)), sampling_method=str(loader_config.get("sampling_method", "permutation_per_epoch")), record_step_loss=str(training_config.get("record_step_loss", "none")), record_first_step_evaluation=bool(training_config.get("record_first_step_evaluation", False)), record_epoch_evaluation=bool(training_config.get("record_epoch_evaluation", False)), record_step_evaluation_interval=training_config.get("record_step_evaluation_interval"), record_first_validation_evaluation=bool(training_config.get("record_first_validation_evaluation", False)), record_step_validation_interval=training_config.get("record_step_validation_interval"), profiling_config=profiling_config_from_mapping(_mapping(config, "profiling")), callbacks=[_Callback(context)], on_epoch_checkpoint=save_evaluation_checkpoint)
        if (resume := checkpoint_config.get("resume")):
            load_epoch_checkpoint(path=str(resume), model=model, optimizer=optimizer, trainer=trainer, config_digest=config_digest)
        seed_batch_order(backend, streams)
        trainer.fit(x_train, t_train, x_valid, t_valid)
        trainer.dump_profiling_artifacts(Path(str(context.metadata["artifact_root"])) / "profiles")
        profiling = trainer.profiling_metrics()
        metrics = build_final_metrics(
            train_loss=float(trainer.losses.train[-1]), test_loss=float(epoch_test_metrics[-1]["loss"]),
            train_accuracy=float(trainer.accuracies.train[-1]), test_accuracy=float(epoch_test_metrics[-1]["accuracy"]),
            profiling_metrics=profiling, total_updates=trainer.global_step,
            completed_epochs=int(training_config.get("max_epochs", 1)), samples_seen=len(x_train) * int(training_config.get("max_epochs", 1)),
        )
        history = update_history(train_logs=trainer.logs.train)
        history += [("update", step, "train/raw_loss", loss) for step, loss in trainer.step_losses]
        history += [("book_epoch", step, key, value) for step, metrics in trainer.graph_evaluations for key, value in metrics.items()]
        legacy_evaluations = evaluation_history(valid_logs=trainer.logs.valid)
        history += legacy_evaluations
        history += [("eval", eval_step, metric, value) for eval_step, global_step, metrics in trainer.validation_evaluations for metric, value in {**metrics, "global_update": float(global_step)}.items()]
        history += epoch_history(train_losses=trainer.losses.train, test_losses=[], train_accuracies=trainer.accuracies.train, test_accuracies=[])
        history += [("epoch", index, "valid/loss", float(value)) for index, value in enumerate(trainer.losses.valid)]
        history += [("epoch", index, "valid/accuracy", float(value)) for index, value in enumerate(trainer.accuracies.valid)]
        history += [("epoch", index, f"test/{metric}", float(value)) for index, values in enumerate(epoch_test_metrics) for metric, value in values.items()]
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


def _optimizer(config: dict[str, object], params):
    name = str(config.get("name", "sgd"))
    learning_rate = float(config.get("learning_rate", 0.01))
    weight_decay = float(config.get("weight_decay", 0.0))
    hooks = [L2Regularization(weight_decay)] if weight_decay else None
    if name == "sgd": return SGD(params, lr=learning_rate, pre_step_hooks=hooks)
    if name == "momentum": return Momentum(params, lr=learning_rate, momentum=float(config.get("momentum", 0.9)), pre_step_hooks=hooks)
    if name == "adagrad": return AdaGrad(params, lr=learning_rate, eps=float(config.get("eps", 1e-7)), pre_step_hooks=hooks)
    if name == "adam": return Adam(params, lr=learning_rate, beta1=float(config.get("beta1", 0.9)), beta2=float(config.get("beta2", 0.999)), eps=float(config.get("eps", 1e-7)), pre_step_hooks=hooks)
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


def _apply_input_transform(*, dataset: dict[str, object], backend, x_train, x_test, artifact_root: Path) -> dict[str, object]:
    """Create deterministic, data-only transforms without consuming training RNG streams."""
    transform = dataset.get("input_transform", {"name": "identity"})
    if not isinstance(transform, dict):
        raise ValueError("dataset.input_transform must be a mapping")
    name = str(transform.get("name", "identity"))
    if name == "identity":
        return {"name": name}
    if name != "pixel_permutation":
        raise ValueError(f"unknown dataset input transform: {name}")
    if len(x_train.shape) not in {2, 4} or len(x_test.shape) != len(x_train.shape):
        raise ValueError("pixel_permutation requires flat MNIST or NCHW image tensors")
    feature_count = int(np.prod(x_train.shape[1:]))
    if feature_count != int(np.prod(x_test.shape[1:])):
        raise ValueError("train and test tensors must have the same feature count")
    seed = int(transform["seed"])
    permutation = np.random.default_rng(seed).permutation(feature_count).astype(np.int64)
    digest = hashlib.sha256(permutation.tobytes()).hexdigest()
    target = artifact_root / "data" / "pixel_permutation.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, permutation)
    return {
        "name": name,
        "seed": seed,
        "feature_count": feature_count,
        "sha256": digest,
        "artifact": str(target.relative_to(artifact_root)),
        "permutation": backend.xp.asarray(permutation),
    }


def _permute_pixels(x, permutation):
    original_shape = x.shape
    return x.reshape(len(x), -1)[:, permutation].reshape(original_shape)


def _validation_probe(*, dataset: dict[str, object], backend, x_train, t_train, artifact_root: Path):
    size = int(dataset.get("validation_size", 0))
    if size == 0:
        return x_train, t_train, None, None, {"size": 0}
    if not 0 < size < len(x_train):
        raise ValueError("dataset.validation_size must be between 1 and train size - 1")
    seed = int(dataset.get("validation_seed", 0))
    indices = np.random.default_rng(seed).permutation(len(x_train))
    valid_indices, train_indices = indices[:size], indices[size:]
    target = artifact_root / "data" / "validation_indices.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, valid_indices)
    xp = backend.xp
    return (x_train[xp.asarray(train_indices)], t_train[xp.asarray(train_indices)], x_train[xp.asarray(valid_indices)], t_train[xp.asarray(valid_indices)], {"size": size, "seed": seed, "artifact": str(target.relative_to(artifact_root)), "sha256": hashlib.sha256(valid_indices.tobytes()).hexdigest()})
