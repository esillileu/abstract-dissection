from __future__ import annotations

# ruff: noqa: E701, E702

from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import time

import numpy as np

from mlprosection.datasets import load_mnist
from mlprosection.datasets.mnist import save_file as mnist_cache_path
from mlprosection.nn.layers import BatchNormalization, SoftmaxWithLoss
from mlprosection.nn.model import DeepCNN, MLP, SimpleCNN
from mlprosection.optim.SGD import AdaGrad, Adam, Momentum, SGD
from mlprosection.optim.transform import L2Regularization
from mlprosection.trainer import ForwardTrainer
from mlprosection.events import EpochEvent, TrainEndEvent, TrainingWindowEvent, UpdateEvent

from mlprosection.experiment.checkpoint import load_epoch_checkpoint, save_epoch_checkpoint
from mlprosection.experiment.contracts import ExperimentResult
from mlprosection.experiment.event_executor import EvaluationRequest, EventExperimentExecutor
from mlprosection.experiment.executor import ExperimentContext
from mlprosection.experiment.metrics import build_final_metrics
from mlprosection.experiment.registry import register_executor
from mlprosection.experiment.reproducibility import configure_runtime, seed_batch_order

from .records import DS1Records


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
        x_train_probe, t_train_probe, train_probe_metadata = _train_evaluation_probe(
            dataset=dataset,
            training=training_config,
            backend=backend,
            x_train=x_train,
            t_train=t_train,
            artifact_root=Path(str(context.metadata["artifact_root"])),
        )
        context.metadata["data"] = {
            "cache_path": mnist_cache_path,
            "cache_sha256": _file_digest(Path(mnist_cache_path)),
            "train_samples": len(x_train), "test_samples": len(x_test), "flatten": flatten,
            "input_transform": transform_metadata, "validation": validation_metadata,
            "train_evaluation": train_probe_metadata,
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
        trainer_holder: dict[str, ForwardTrainer] = {}
        evaluation_config = _mapping(config, "evaluation")
        valid_request = None if x_valid is None or t_valid is None else EvaluationRequest(
            "mnist-valid", "valid", (x_valid, t_valid), ("loss", "accuracy"),
        )
        train_request = None if x_train_probe is None or t_train_probe is None else EvaluationRequest(
            "mnist-train-probe", "train", (x_train_probe, t_train_probe), ("loss", "accuracy"),
        )
        test_request = EvaluationRequest("mnist-test-full", "test", (x_test, t_test), ("loss", "accuracy"))
        test_probe_size = evaluation_config.get("test_probe_size")
        test_probe_request = test_request if test_probe_size is None else EvaluationRequest(
            "mnist-test-probe", "test", (x_test[:int(test_probe_size)], t_test[:int(test_probe_size)]), ("loss", "accuracy"),
        )
        schedule = _mapping(evaluation_config, "schedule")
        request_by_set = {
            "valid": valid_request,
            "train-first-300": train_request,
            "train-first-1000": train_request,
            "test-full": test_request,
            "test-first-1000": test_probe_request,
        }
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
                should = bool(update_spec.get("first", False) and event.update == 1)
                should |= every is not None and event.update % int(every) == 0
                if should:
                    return scheduled_requests(update_spec)
            first_epoch_spec = schedule.get("on_epoch_first_update")
            if isinstance(first_epoch_spec, dict) and event.epoch not in seen_epochs:
                seen_epochs.add(event.epoch)
                return scheduled_requests(first_epoch_spec)
            interval = training_config.get("record_step_validation_interval")
            should = bool(training_config.get("record_first_validation_evaluation", False) and event.update == 1)
            should |= interval is not None and event.update % int(interval) == 0
            if not should:
                return ()
            return tuple(request for request in (valid_request, train_request if bool(training_config.get("record_step_train_evaluation", False)) else None) if request is not None)
        def after_epoch(event):
            if bool(checkpoint_config.get("save_on_eval", False)):
                path = save_epoch_checkpoint(root=Path(str(context.metadata["checkpoint_root"])), model=model, optimizer=optimizer, trainer=trainer_holder["trainer"], config_digest=config_digest)
                callback = context.metadata.get("record_eval_checkpoint")
                if callable(callback):
                    callback(path)
        events = EventExperimentExecutor(
            records=DS1Records(), evaluate=evaluate_request, update_requests=update_requests,
            epoch_requests=lambda _event: scheduled_requests(schedule.get("on_epoch_end")),
            terminal_requests=lambda _event: scheduled_requests(schedule.get("on_train_end")),
            after_epoch=after_epoch,
        )
        trainer = ForwardTrainer(
            model,
            SoftmaxWithLoss().to(model.backend),
            optimizer,
            max_epochs=int(training_config.get("max_epochs", 1)),
            max_updates=None if max_updates is None else int(max_updates),
            batch_size=int(loader_config.get("batch_size", 32)),
            drop_last=bool(loader_config.get("drop_last", False)),
            sampling_method=str(loader_config.get("sampling_method", "permutation_per_epoch")),
            event_receivers=[events],
        )
        trainer_holder["trainer"] = trainer
        if (resume := checkpoint_config.get("resume")):
            load_epoch_checkpoint(path=str(resume), model=model, optimizer=optimizer, trainer=trainer, config_digest=config_digest)
        seed_batch_order(backend, streams)
        records = events.run(lambda: trainer.fit(x_train, t_train), start_update=trainer.global_step + 1)
        records.write_csv(Path(str(context.metadata["artifact_root"])))
        final_train = trainer.evaluate(x_train, t_train)
        final_test = trainer.evaluate(x_test, t_test)
        profiling: dict[str, int | float] = {}
        metrics = build_final_metrics(
            train_loss=final_train.loss, test_loss=final_test.loss,
            train_accuracy=final_train.accuracy, test_accuracy=final_test.accuracy,
            profiling_metrics=profiling, total_updates=trainer.global_step,
            completed_epochs=trainer.epoch, samples_seen=sum(int(row["batch_size"]) for row in records.updates),
        )
        history = list(records.history_rows())
        return ExperimentResult(metrics=metrics, artifact_root=_artifact_root(config), model=model, history=tuple(history), profiling_metrics=profiling)


class _SupervisedEvents:
    """Executor-owned DS1 schedule adapter for the event-based trainer."""

    def __init__(self, *, context, training, x_valid, t_valid, x_train_probe, t_train_probe, x_test, t_test, checkpoint_config, checkpoint_identity, model, optimizer):
        self.context, self.training = context, training
        self.x_valid, self.t_valid = x_valid, t_valid
        self.x_train_probe, self.t_train_probe = x_train_probe, t_train_probe
        self.x_test, self.t_test = x_test, t_test
        self.checkpoint_config, self.checkpoint_identity = checkpoint_config, checkpoint_identity
        self.model, self.optimizer, self.trainer = model, optimizer, None
        self.updates: list[UpdateEvent] = []
        self.history: list[tuple[str, int, str, float]] = []
        self.epoch_test_metrics = []
        self.timing_windows: list[TrainingWindowEvent] = []
        self._window_start_update: int | None = None
        self._window_started_ns: int | None = None

    def begin_timing_window(self, *, start_update: int) -> None:
        self._window_start_update = start_update
        self._window_started_ns = time.perf_counter_ns()

    def on_update(self, event: UpdateEvent) -> None:
        self.updates.append(event)
        interval = self.training.get("record_step_validation_interval")
        should_evaluate = bool(self.training.get("record_first_validation_evaluation", False) and event.update == 1)
        should_evaluate |= interval is not None and event.update % int(interval) == 0
        if should_evaluate:
            self._close_timing_window(event, closed_by="probe", evaluate=True)

    def on_epoch(self, event: EpochEvent) -> None:
        assert self.trainer is not None
        if self._window_start_update is not None and self._window_start_update <= event.end_update:
            self._close_timing_window(event, closed_by="epoch_end", evaluate=False)
        values = self.trainer.evaluate(self.x_test, self.t_test)
        self.epoch_test_metrics.append(values)
        self.history.extend(("epoch", event.epoch, f"test/{name}", float(value)) for name, value in {"loss": values.loss, "accuracy": values.accuracy}.items() if value is not None)
        if bool(self.checkpoint_config.get("save_on_eval", False)):
            path = save_epoch_checkpoint(root=Path(str(self.context.metadata["checkpoint_root"])), model=self.model, optimizer=self.optimizer, trainer=self.trainer, config_digest=self.checkpoint_identity)
            callback = self.context.metadata.get("record_eval_checkpoint")
            if callable(callback):
                callback(path)

    def on_train_end(self, event: TrainEndEvent) -> None:
        if self._window_start_update is not None and self._window_start_update <= event.update:
            self._close_timing_window(event, closed_by="terminal", evaluate=False)

    def _evaluate_probe(self, event: UpdateEvent) -> None:
        assert self.trainer is not None
        if self.x_valid is not None and self.t_valid is not None:
            self._append_evaluation(event.update, "valid", self.trainer.evaluate(self.x_valid, self.t_valid))
        if bool(self.training.get("record_step_train_evaluation", False)):
            if self.x_train_probe is None or self.t_train_probe is None:
                raise ValueError("step train evaluation requires a train probe")
            self._append_evaluation(event.update, "train", self.trainer.evaluate(self.x_train_probe, self.t_train_probe))

    def _append_evaluation(self, step: int, split: str, values) -> None:
        for name, value in {"loss": values.loss, "accuracy": values.accuracy}.items():
            if value is not None:
                self.history.append(("eval", step, f"{split}/{name}", float(value)))

    def _close_timing_window(self, event, *, closed_by, evaluate: bool) -> None:
        assert self._window_start_update is not None
        assert self._window_started_ns is not None
        train_wall_time_ns = time.perf_counter_ns() - self._window_started_ns
        eval_started_ns = time.perf_counter_ns()
        if evaluate:
            self._evaluate_probe(event)
        eval_wall_time_ns = time.perf_counter_ns() - eval_started_ns if evaluate else None
        end_update = self._event_update(event)
        self.timing_windows.append(
            TrainingWindowEvent(
                start_update=self._window_start_update,
                end_update=end_update,
                update_count=end_update - self._window_start_update + 1,
                closed_by=closed_by,
                train_wall_time_ns=train_wall_time_ns,
                eval_wall_time_ns=eval_wall_time_ns,
            )
        )
        self.begin_timing_window(start_update=end_update + 1)

    @staticmethod
    def _event_update(event) -> int:
        if isinstance(event, UpdateEvent | TrainEndEvent):
            return event.update
        return event.end_update


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


def _train_evaluation_probe(*, dataset: dict[str, object], training: dict[str, object], backend, x_train, t_train, artifact_root: Path):
    if not bool(training.get("record_step_train_evaluation", False)):
        return None, None, {"enabled": False, "size": 0}
    size = int(dataset.get("train_evaluation_size", 1_000))
    if not 0 < size <= len(x_train):
        raise ValueError("dataset.train_evaluation_size must be between 1 and train size")
    seed = int(dataset.get("train_evaluation_seed", 0))
    indices = np.random.default_rng(seed).permutation(len(x_train))[:size]
    target = artifact_root / "data" / "train_evaluation_indices.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, indices)
    xp = backend.xp
    return (x_train[xp.asarray(indices)], t_train[xp.asarray(indices)], {"enabled": True, "size": size, "seed": seed, "artifact": str(target.relative_to(artifact_root)), "sha256": hashlib.sha256(indices.tobytes()).hexdigest()})
