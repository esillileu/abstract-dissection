from __future__ import annotations

# ruff: noqa: E701, E702

from pathlib import Path
from dataclasses import asdict
import csv
import hashlib
import json

import numpy as np

from mlprosection.datasets import load_mnist
from mlprosection.datasets.mnist import save_file as mnist_cache_path
from mlprosection.nn.layers import BatchNormalization
from mlprosection.nn.model.architecture import DeepCNN, MLP, SimpleCNN
from mlprosection.nn.objective import SoftmaxCrossEntropy
from mlprosection.optim.SGD import AdaGrad, Adam, Momentum, SGD
from mlprosection.optim.transform import L2Regularization
from mlprosection.trainer import ForwardTrainer

from mlprosection.experiment.checkpoint import (
    CheckpointManager,
    CheckpointRetentionPolicy,
    load_epoch_checkpoint,
)
from mlprosection.experiment.contracts import ExperimentResult
from mlprosection.experiment.event_executor import EvaluationRequest, EventExperimentExecutor
from mlprosection.experiment.executor import ExperimentContext
from mlprosection.experiment.metrics import build_final_metrics
from mlprosection.experiment.profiling import create_runtime_monitor, training_summary
from mlprosection.experiment.registry import register_executor
from mlprosection.experiment.reproducibility import configure_runtime, seed_batch_order
from mlprosection.profiling.backend import create_device_timer

from .records import DS1Records


def get_observation_executor(config: dict[str, object]):
    group_id = str(config.get("execution_group_id", ""))
    if group_id == "GO01":
        return OptimizerTrajectoryObservationExecutor()
    if group_id == "GO02":
        return ActivationObservationExecutor()
    raise ValueError(f"unknown DS1 observation group: {group_id}")


@register_executor("supervised_classification")
class SupervisedClassificationExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        dataset, model_config, objective_config, training_config, loader_config, optimizer_config = (_mapping(config, key) for key in ("dataset", "model", "objective", "training", "loader", "optimizer"))
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
            "evaluation_sources": _mapping(config, "evaluation").get("sources", ()),
        }
        model = _model(model_config)
        objective = _objective(objective_config, model.backend)
        if any(isinstance(layer, BatchNormalization) for layer in model.children()):
            model.forward(x_train[:1])
        optimizer = _optimizer(
            optimizer_config,
            _training_parameters(model, objective),
        )
        checkpoint_config = _mapping(config, "checkpoint")
        checkpoint_identity = dict(config)
        checkpoint_identity["checkpoint"] = dict(checkpoint_config)
        checkpoint_identity["checkpoint"].pop("resume", None)
        config_digest = hashlib.sha256(json.dumps(checkpoint_identity, sort_keys=True, default=str).encode()).hexdigest()
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
        seen_epochs: set[int] = set()

        def scheduled_requests(spec: object):
            if not isinstance(spec, dict):
                return ()
            sets = spec.get("sets", ())
            if not isinstance(sets, list | tuple):
                raise ValueError("evaluation schedule sets must be a list")
            return tuple(request_by_set[str(name)] for name in sets if request_by_set.get(str(name)) is not None)

        def evaluate_request(request):
            return trainer_holder["trainer"].evaluate(*request.source, metrics=request.metrics)
        def update_requests(event):
            update_spec = schedule.get("on_update")
            if isinstance(update_spec, dict):
                every = update_spec.get("every")
                start = int(update_spec.get("start", 1 if update_spec.get("first", False) else int(every or event.update)))
                stop = update_spec.get("stop")
                if stop is not None and event.update > int(stop):
                    return ()
                should = event.update == start
                should |= every is not None and event.update >= start and (event.update - start) % int(every) == 0
                if should:
                    return scheduled_requests(update_spec)
            first_epoch_spec = schedule.get("on_epoch_first_update")
            if isinstance(first_epoch_spec, dict) and event.epoch not in seen_epochs:
                seen_epochs.add(event.epoch)
                return scheduled_requests(first_epoch_spec)
            return ()
        def after_epoch(event):
            records_sink.flush()
            manager = checkpoint_manager_holder["manager"]
            manager.save_latest()
            periodic = manager.save_periodic_if_due()
            if bool(checkpoint_config.get("save_on_eval", False)) and periodic is not None:
                callback = context.metadata.get("record_eval_checkpoint")
                if callable(callback):
                    callback(periodic.path)
        artifact_root = Path(str(context.metadata["artifact_root"]))
        records_sink = DS1Records()
        records_sink.bind_artifact_root(artifact_root)
        monitor = create_runtime_monitor(backend, _mapping(config, "profiling"))
        events = EventExperimentExecutor(
            records=records_sink, evaluate=evaluate_request, update_requests=update_requests,
            epoch_requests=lambda _event: scheduled_requests(schedule.get("on_epoch_end")),
            terminal_requests=lambda _event: scheduled_requests(schedule.get("on_train_end")),
            after_epoch=after_epoch,
            device_timer=_device_timer(config, backend),
            progress=context.metadata.get("progress_reporter"),
        )
        trainer = ForwardTrainer(
            model,
            objective,
            optimizer,
            max_epochs=int(training_config.get("max_epochs", 1)),
            max_updates=None if max_updates is None else int(max_updates),
            batch_size=int(loader_config.get("batch_size", 32)),
            drop_last=bool(loader_config.get("drop_last", False)),
            sampling_method=str(loader_config.get("sampling_method", "permutation_per_epoch")),
            event_receivers=[events],
        )
        trainer_holder["trainer"] = trainer
        checkpoint_manager = CheckpointManager(
            root=Path(str(context.metadata["checkpoint_root"])),
            model=model,
            objective=objective,
            optimizer=optimizer,
            trainer=trainer,
            config_digest=config_digest,
            policy=CheckpointRetentionPolicy.from_mapping(checkpoint_config),
        )
        checkpoint_manager_holder["manager"] = checkpoint_manager
        if (resume := checkpoint_config.get("resume")):
            load_epoch_checkpoint(path=str(resume), model=model, objective=objective, optimizer=optimizer, trainer=trainer, config_digest=config_digest)
        seed_batch_order(backend, streams)
        with training_summary(monitor):
            records = events.run(lambda: trainer.fit(x_train, t_train), start_update=trainer.global_step + 1)
        latest = checkpoint_manager.current("latest")
        if latest is not None:
            records.add_checkpoint(
                update=latest.update, epoch=latest.epoch, kind="latest",
                path=latest.path, sha256=latest.sha256,
            )
        best = checkpoint_manager.current("best")
        if best is not None:
            records.add_checkpoint(
                update=best.update, epoch=best.epoch, kind="selected",
                path=best.path, sha256=best.sha256,
            )
        for periodic in checkpoint_manager.retained_periodic():
            records.add_checkpoint(
                update=periodic.update, epoch=periodic.epoch, kind="periodic",
                path=periodic.path, sha256=periodic.sha256,
            )
        records.flush()
        final_train = trainer.evaluate(x_train, t_train)
        final_test = trainer.evaluate(x_test, t_test)
        profiling = monitor.metrics()
        metrics = build_final_metrics(
            train_loss=final_train.loss, test_loss=final_test.loss,
            train_accuracy=final_train.accuracy, test_accuracy=final_test.accuracy,
            profiling_metrics=profiling, total_updates=trainer.global_step,
            completed_epochs=trainer.epoch, samples_seen=sum(int(row["batch_size"]) for row in records.updates),
        )
        return ExperimentResult(
            metrics=metrics,
            artifact_root=_artifact_root(config),
            model=model,
            metric_rows=records.mlflow_metric_rows(),
            profiling_metrics=profiling,
        )

class OptimizerTrajectoryObservationExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        artifact_root = Path(str(context.metadata.get("artifact_root", _artifact_root(config))))
        observations = artifact_root / "observations"
        observations.mkdir(parents=True, exist_ok=True)
        optimizer = str(_mapping(config, "optimizer").get("name", "toy_sgd"))
        lr = float(_mapping(config, "optimizer").get("learning_rate", 0.95))
        max_updates = int(_mapping(config, "training").get("max_updates", 30))
        x, y = -7.0, 2.0
        vx = vy = gx2 = gy2 = mx = my = ux = uy = 0.0
        beta1, beta2, eps, momentum = 0.9, 0.999, 1e-7, 0.9
        rows = []
        for update in range(max_updates):
            objective = x * x / 20.0 + y * y
            grad_x, grad_y = x / 10.0, 2.0 * y
            rows.append({"update": update, "x": x, "y": y, "objective": objective, "grad_x": grad_x, "grad_y": grad_y})
            name = optimizer.lower().removeprefix("toy_").replace("-", "_")
            if name == "momentum":
                vx = momentum * vx - lr * grad_x
                vy = momentum * vy - lr * grad_y
                x += vx
                y += vy
            elif name == "adagrad":
                gx2 += grad_x * grad_x
                gy2 += grad_y * grad_y
                x -= lr * grad_x / (np.sqrt(gx2) + eps)
                y -= lr * grad_y / (np.sqrt(gy2) + eps)
            elif name == "adam":
                step = update + 1
                mx = beta1 * mx + (1 - beta1) * grad_x
                my = beta1 * my + (1 - beta1) * grad_y
                ux = beta2 * ux + (1 - beta2) * grad_x * grad_x
                uy = beta2 * uy + (1 - beta2) * grad_y * grad_y
                x -= lr * (mx / (1 - beta1 ** step)) / (np.sqrt(ux / (1 - beta2 ** step)) + eps)
                y -= lr * (my / (1 - beta1 ** step)) / (np.sqrt(uy / (1 - beta2 ** step)) + eps)
            elif name == "sgd":
                x -= lr * grad_x
                y -= lr * grad_y
            else:
                raise ValueError(f"unknown toy optimizer: {optimizer}")
        _write_rows(observations / "trajectory.csv", rows, ["update", "x", "y", "objective", "grad_x", "grad_y"])
        records = DS1Records()
        records.bind_artifact_root(artifact_root)
        records.flush()
        return ExperimentResult(
            metrics={"final/status/success": 1.0, "final/system/total_updates": float(max_updates), "final/system/completed_epochs": 0.0, "final/system/samples_seen": 0.0},
            artifact_root=artifact_root,
            metric_rows=tuple((int(row["update"]), f"update/trajectory/{key}", float(row[key])) for row in rows for key in ("x", "y", "objective", "grad_x", "grad_y")),
        )


class ActivationObservationExecutor:
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult:
        artifact_root = Path(str(context.metadata.get("artifact_root", _artifact_root(config))))
        observations = artifact_root / "observations"
        observations.mkdir(parents=True, exist_ok=True)
        model_config = _mapping(config, "model")
        seed = int(_mapping(config, "dataset").get("input_seed", 40402))
        model_seed = int(model_config.get("model_seed", 40403))
        rng = np.random.default_rng(seed)
        weight_rng = np.random.default_rng(model_seed)
        x = rng.normal(size=(1000, 100))
        activation = str(model_config.get("activation", "relu"))
        initializer = str(model_config.get("initializer", "he"))
        hist_rows = []
        summary_rows = []
        for layer in range(1, int(model_config.get("depth", 5)) + 1):
            w = weight_rng.normal(scale=_initializer_scale(initializer, x.shape[1]), size=(x.shape[1], int(model_config.get("width", 100))))
            x = _activation(x @ w, activation)
            hist, edges = np.histogram(x, bins=50)
            for index, count in enumerate(hist):
                hist_rows.append({"layer": layer, "bin_index": index, "bin_left": edges[index], "bin_right": edges[index + 1], "count": int(count), "sample_count": int(x.size)})
            summary_rows.append({"layer": layer, "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max()), "zero_ratio": float((x == 0).mean()), "sample_count": int(x.size)})
        _write_rows(observations / "activation_histogram.csv", hist_rows, ["layer", "bin_index", "bin_left", "bin_right", "count", "sample_count"])
        _write_rows(observations / "activation_summary.csv", summary_rows, ["layer", "mean", "std", "min", "max", "zero_ratio", "sample_count"])
        records = DS1Records()
        records.bind_artifact_root(artifact_root)
        records.flush()
        return ExperimentResult(
            metrics={"final/status/success": 1.0, "final/system/total_updates": 0.0, "final/system/completed_epochs": 0.0, "final/system/samples_seen": 1000.0},
            artifact_root=artifact_root,
            metric_rows=tuple((0, f"observation/activation/layer_{row['layer']}/{key}", float(row[key])) for row in summary_rows for key in ("mean", "std", "zero_ratio")),
        )


def _device_timer(config: dict[str, object], backend):
    profiling = _mapping(config, "profiling")
    return create_device_timer(backend, enabled=bool(profiling.get("device_timing", False)))


def _model(config: dict[str, object]):
    name = str(config.get("name", "MLP"))
    values = {key: value for key, value in config.items() if key not in {"name", "family", "task_type", "input_shape", "output_shape", "structure_signature", "use_batchnorm", "use_dropout", "num_hidden_layers", "num_conv_layers", "normalization", "model/flops", "model/macs"}}
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
    raise ValueError(f"unknown model name: {name}")


def _objective(config: dict[str, object], backend):
    name = str(config.get("name", "SoftmaxCrossEntropy"))
    if name == "SoftmaxCrossEntropy":
        return SoftmaxCrossEntropy(
            reduction=str(config.get("reduction", "mean")),
            backend=backend,
        )
    raise ValueError(f"unknown objective name: {name}")


def _training_parameters(model, objective):
    return [
        *((f"model.{name}", parameter) for name, parameter in model.named_parameters()),
        *((f"objective.{name}", parameter) for name, parameter in objective.named_parameters()),
    ]


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
    return Path(str(run.get("artifact_root", "exp/ds1/results/runs"))) / str(run.get("name", config["kind"]))


def _evaluation_requests(*, evaluation: dict[str, object], x_train, t_train, x_valid, t_valid, x_test, t_test) -> dict[str, EvaluationRequest]:
    raw_sources = evaluation.get("sources", ())
    if not isinstance(raw_sources, list | tuple):
        raise ValueError("evaluation.sources must be a list")
    requests: dict[str, EvaluationRequest] = {}
    for source in raw_sources:
        if not isinstance(source, dict):
            raise ValueError("evaluation source must be a mapping")
        source_id = str(source["id"])
        split = str(source["split"])
        kind = str(source["kind"])
        if split == "train":
            x, t = x_train, t_train
        elif split == "valid":
            if x_valid is None or t_valid is None:
                continue
            x, t = x_valid, t_valid
        elif split == "test":
            x, t = x_test, t_test
        else:
            raise ValueError(f"unsupported evaluation source split: {split}")
        if kind == "first_n":
            count = int(source["count"])
            x, t = x[:count], t[:count]
        elif kind != "full":
            raise ValueError(f"unsupported evaluation source kind: {kind}")
        requests[source_id] = EvaluationRequest(source_id, split, (x, t), ("loss", "accuracy"))
    return requests


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    if path.is_file():
        return _file_digest(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(_file_digest(child).encode())
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _initializer_scale(initializer: str, fan_in: int) -> float:
    if initializer == "he":
        return float(np.sqrt(2.0 / fan_in))
    if initializer == "xavier":
        return float(np.sqrt(1.0 / fan_in))
    if initializer.startswith("std:"):
        return float(initializer.split(":", 1)[1])
    return 1.0


def _activation(value: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(0.0, value)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-value))
    if name == "tanh":
        return np.tanh(value)
    raise ValueError(f"unknown activation: {name}")


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
