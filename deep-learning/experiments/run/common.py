from __future__ import annotations

# ruff: noqa: E402

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlprosection import Tensor
from mlprosection.core.backend import get_default_backend
from mlprosection.datasets import load_mnist
from mlprosection.nn.initailizer import he_normal_, xavier_normal_
from mlprosection.nn.layers import (
    Affine,
    BatchNormalization,
    Conv2D,
    Dropout,
    Flatten,
    Layer,
    MaxPool2D,
    Relu,
    Sigmoid,
    SoftmaxWithLoss,
)
from mlprosection.nn.model.sequential import Sequential
from mlprosection.optim.SGD import AdaGrad, Adam, Momentum, SGD
from mlprosection.optim.transform import L2Regularization
from mlprosection.profiling import ProfilingConfig
from mlprosection.tracking import (
    MLflowRunLogger,
    RunIdentity,
    make_condition_key,
    make_run_key,
)
from mlprosection.tracking.artifacts import (
    current_git_info,
    environment_artifacts,
    file_digest,
    parameter_manifest,
    pip_freeze,
    write_git_diff,
    write_history_csv,
    write_json,
    write_memory_history_csv,
    write_runtime_history_csv,
    write_text,
)
from mlprosection.tracking.mlflow_logger import flatten_dict
from mlprosection.tracking.schema import (
    build_epoch_metric_rows,
    build_memory_history_rows,
    build_runtime_history_rows,
    build_schema_metrics,
)
from mlprosection.trainer import ForwardTrainer


SCHEMA_VERSION = 1
RESULT_ROOT = Path("experiments/results/mlflow_artifacts")
CHECKPOINT_ROOT = Path("experiments/params/run")


@dataclass(frozen=True)
class RunSpec:
    atomic_run_id: str
    experiment_ids: tuple[str, ...]
    execution_group_id: str
    recipe_id: str
    structure_signature: str
    dataset: dict[str, object]
    loader: dict[str, object]
    model: dict[str, object]
    initializer: dict[str, object]
    optimizer: dict[str, object]
    scheduler: dict[str, object]
    loss: dict[str, object]
    training: dict[str, object]
    evaluation: dict[str, object]
    numerics: dict[str, object]
    checkpoint: dict[str, object]
    profiling: dict[str, object]
    regularization: dict[str, object] | None = None
    policy: dict[str, object] | None = None


class TrainAwareSequential(Sequential):
    def __init__(self, *layers: Layer) -> None:
        super().__init__(*layers)
        self.train_flg = True

    def forward_manual(self, x):
        for layer in self.layers:
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer.forward(x, train_flg=self.train_flg)
            else:
                x = layer.forward(x)
        return x


def parser(description: str, choices: list[str]) -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=description)
    argument_parser.add_argument("--atomic-run-id", choices=choices, required=True)
    argument_parser.add_argument("--seed", type=int, default=int(os.getenv("MLPROSECTION_MASTER_SEED", "0")))
    argument_parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    argument_parser.add_argument("--mlflow-experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlprosection"))
    argument_parser.add_argument("--no-mlflow", action="store_true")
    return argument_parser


def seed_config(master_seed: int) -> dict[str, int]:
    return {
        "master": master_seed,
        "model_init": master_seed,
        "batch_order": master_seed + 10000,
        "dropout": master_seed + 20000,
        "negative_sampling": master_seed + 30000,
        "synthetic_input": master_seed + 40000,
        "dataset_split": master_seed,
        "worker": master_seed + 50000,
    }


def profiling_config_from_env() -> ProfilingConfig:
    return ProfilingConfig(
        enabled=env_flag("MLPROSECTION_PROFILE", "0"),
        start_step=int(os.getenv("MLPROSECTION_PROFILE_START_STEP", "0")),
        num_steps=int(os.getenv("MLPROSECTION_PROFILE_STEPS", "5")),
        profile_memory=True,
        sample_memory_every_n_steps=int(os.getenv("MLPROSECTION_MEMORY_SAMPLE_EVERY", "20")),
    )


def profiling_config_dict(config: ProfilingConfig) -> dict[str, object]:
    return {
        "enabled": config.enabled,
        "python_enabled": config.profile_python,
        "nsight_enabled": False,
        "warmup_steps": config.start_step,
        "profile_steps": config.num_steps,
        "record_shapes": False,
        "record_memory": config.profile_memory,
        "sample_interval_ms": None,
        "export_python_prof": config.profile_python,
        "export_nsight_report": False,
        "export_summary_json": True,
    }


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def load_mnist_tensors(*, flatten: bool, gpu: bool, train_limit: int | None, test_limit: int | None):
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=flatten, gpu=gpu)
    if train_limit is not None:
        x_train, t_train = x_train[:train_limit], t_train[:train_limit]
    if test_limit is not None:
        x_test, t_test = x_test[:test_limit], t_test[:test_limit]
    return (x_train, t_train), (x_test, t_test)


def make_mlp(
    *,
    input_size: int,
    hidden_size: int,
    hidden_layers: int,
    output_size: int,
    initializer: str,
    activation: str = "relu",
    batchnorm: bool = False,
    dropout_ratio: float | None = None,
) -> TrainAwareSequential:
    layers = []
    in_features = input_size
    for _ in range(hidden_layers):
        affine = Affine(in_features, hidden_size)
        reset_affine_weight(affine, initializer)
        layers.append(affine)
        if batchnorm:
            layers.append(BatchNormalization())
        layers.append(make_activation(activation))
        if dropout_ratio is not None and dropout_ratio > 0:
            layers.append(Dropout(dropout_ratio))
        in_features = hidden_size

    output = Affine(in_features, output_size)
    reset_affine_weight(output, initializer)
    layers.append(output)
    return TrainAwareSequential(*layers)


def make_simple_cnn() -> TrainAwareSequential:
    return TrainAwareSequential(
        Conv2D(1, 30, (5, 5), 1, 0),
        Relu(),
        MaxPool2D(2, 2),
        Flatten(),
        Affine(30 * 12 * 12, 100),
        Relu(),
        Affine(100, 10),
    )


def make_deep_cnn() -> TrainAwareSequential:
    return TrainAwareSequential(
        Conv2D(1, 16, (3, 3), 1, 1),
        Relu(),
        Conv2D(16, 16, (3, 3), 1, 1),
        Relu(),
        MaxPool2D(2, 2),
        Conv2D(16, 32, (3, 3), 1, 1),
        Relu(),
        Conv2D(32, 32, (3, 3), 1, 1),
        Relu(),
        MaxPool2D(2, 2),
        Conv2D(32, 64, (3, 3), 1, 1),
        Relu(),
        Conv2D(64, 64, (3, 3), 1, 1),
        Relu(),
        MaxPool2D(2, 2),
        Flatten(),
        Affine(64 * 3 * 3, 50),
        Relu(),
        Dropout(0.5),
        Affine(50, 10),
    )


def make_activation(name: str) -> Layer:
    if name == "relu":
        return Relu()
    if name == "sigmoid":
        return Sigmoid()
    if name == "tanh":
        return Tanh()
    raise ValueError(f"unknown activation: {name}")


class Tanh(Layer):
    def __init__(self) -> None:
        self.out = None

    def forward_manual(self, x: Tensor) -> Tensor:
        self.out = Tensor(x.backend.xp.tanh(x.data), backend=x.backend)
        return self.out

    def backward_manual(self, dout: Tensor) -> Tensor:
        return Tensor(dout.data * (1 - self.out.data**2), backend=dout.backend)


def reset_affine_weight(layer: Affine, initializer: str) -> None:
    if initializer == "he":
        he_normal_(layer.W)
    elif initializer == "xavier":
        xavier_normal_(layer.W)
    elif initializer.startswith("std:"):
        std = float(initializer.split(":", 1)[1])
        values = layer.backend.xp.random.randn(*layer.W.shape) * std
        layer.W.data[...] = values.astype(layer.W.dtype, copy=False)
    else:
        raise ValueError(f"unknown initializer: {initializer}")


def make_optimizer(name: str, params, *, lr: float, weight_decay: float = 0.0):
    hooks = [L2Regularization(weight_decay)] if weight_decay else None
    if name == "sgd":
        return SGD(params, lr=lr, pre_step_hooks=hooks)
    if name == "momentum":
        return Momentum(params, lr=lr, momentum=0.9, pre_step_hooks=hooks)
    if name == "adagrad":
        return AdaGrad(params, lr=lr, pre_step_hooks=hooks)
    if name == "adam":
        return Adam(params, lr=lr, pre_step_hooks=hooks)
    raise ValueError(f"unknown optimizer: {name}")


def run_classification_trial(
    *,
    spec: RunSpec,
    seed: int,
    model: Layer,
    x_train: Tensor,
    t_train: Tensor,
    x_test: Tensor,
    t_test: Tensor,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    max_epoch: int,
    batch_size: int,
    log_interval: int | None,
    tracking_uri: str,
    mlflow_experiment: str,
    mlflow_enabled: bool,
) -> str:
    get_default_backend().seed(seed)
    criterion = SoftmaxWithLoss().to(model.backend)
    if any(isinstance(layer, BatchNormalization) for layer in model.children()):
        model.forward(x_train[:1])
    optimizer = make_optimizer(
        optimizer_name,
        list(model.named_parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    profiling_config = profiling_config_from_env()
    trainer = ForwardTrainer(
        model,
        criterion,
        optimizer,
        max_epoch=max_epoch,
        batch_size=batch_size,
        log_interval=log_interval,
        profiling_config=profiling_config,
    )
    started = time.perf_counter()
    trainer.fit(x_train, t_train, x_test, t_test)
    run_wall_total_s = time.perf_counter() - started
    profiling_metrics = trainer.profiling_metrics()
    final_metrics = build_schema_metrics(
        train_loss=last_or_none(trainer.losses.train),
        test_loss=last_or_none(trainer.losses.valid),
        train_accuracy=last_or_none(trainer.accuracies.train),
        test_accuracy=last_or_none(trainer.accuracies.valid),
        profiling_metrics=profiling_metrics,
        total_updates=trainer.global_step,
        completed_epochs=max_epoch,
        samples_seen=len(x_train) * max_epoch,
    )
    final_metrics["runtime/run_wall_total_s"] = run_wall_total_s
    for name in ("model/flops", "model/macs"):
        if name in spec.model:
            final_metrics[name] = float(spec.model[name])
    add_classification_summaries(final_metrics, trainer)
    add_inference_metrics(final_metrics, model, x_test)

    epoch_rows = build_epoch_metric_rows(
        train_losses=trainer.losses.train,
        test_losses=trainer.losses.valid,
        train_accuracies=trainer.accuracies.train,
        test_accuracies=trainer.accuracies.valid,
        profiling_metrics=profiling_metrics,
    )
    return log_trial(
        spec=spec,
        seed=seed,
        tracking_uri=tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_enabled=mlflow_enabled,
        final_metrics=final_metrics,
        history_rows=epoch_rows,
        profiling_metrics=profiling_metrics,
        model=model,
        checkpoint_model=model,
    )


def log_trial(
    *,
    spec: RunSpec,
    seed: int,
    tracking_uri: str,
    mlflow_experiment: str,
    mlflow_enabled: bool,
    final_metrics: dict[str, float],
    history_rows: list[tuple[str, int, str, float]],
    profiling_metrics: dict[str, int | float],
    model: Layer | None = None,
    checkpoint_model: Layer | None = None,
) -> str:
    git_info = current_git_info(spec.training.get("entrypoint", "experiments/run"))
    seeds = seed_config(seed)
    condition_config = build_condition_config(spec, git_info)
    identity = build_identity(spec, condition_config, seeds)
    artifact_root = RESULT_ROOT / identity.run_key

    if checkpoint_model is not None:
        checkpoint_path = CHECKPOINT_ROOT / f"{identity.run_key}.npz"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_model.save_params_npz(checkpoint_path)
        checkpoint_digest = file_digest(checkpoint_path)
    else:
        checkpoint_path = None
        checkpoint_digest = None

    write_trial_artifacts(
        artifact_root=artifact_root,
        identity=identity,
        spec=spec,
        condition_config=condition_config,
        seed_config_value=seeds,
        git_info=git_info,
        model=model,
        final_metrics=final_metrics,
        history_rows=history_rows,
        profiling_metrics=profiling_metrics,
        checkpoint_path=checkpoint_path,
        checkpoint_digest=checkpoint_digest,
    )

    if mlflow_enabled:
        logger = MLflowRunLogger(
            tracking_uri=tracking_uri,
            experiment_name=mlflow_experiment,
        )
        tags = build_tags(identity, spec, git_info, model)
        params = flatten_dict(
            {
                **condition_config,
                "seed": seeds,
                "policy": spec.policy or {},
                "regularization": spec.regularization or {},
            }
        )
        logger.start_child(
            run_name=f"{identity.atomic_run_id}-s{identity.master_seed:02d}",
            tags=tags,
            params=params,
        )
        for step_type, step, metric, value in history_rows:
            if step_type == "update":
                logger.log_update_metrics(step, {f"update/{metric}": value})
            elif step_type == "step":
                logger.log_update_metrics(step, {f"step/{metric}": value})
            elif step_type == "epoch":
                logger.log_epoch_metrics(step, {f"epoch/{metric}": value})
        artifact_start = time.perf_counter()
        logger.log_final_metrics(final_metrics)
        logger.log_artifact_tree(artifact_root)
        artifact_total = time.perf_counter() - artifact_start
        logger.log_final_metrics({"runtime/artifact_logging_total_s": artifact_total})
        logger.finalize_success()

    return identity.run_key


def build_condition_config(spec: RunSpec, git_info: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "atomic_run_id": spec.atomic_run_id,
        "execution_group_id": spec.execution_group_id,
        "recipe_id": spec.recipe_id,
        "structure_signature": spec.structure_signature,
        "code": {
            "git_commit": git_info["commit"],
            "entrypoint": git_info["entrypoint"],
        },
        "dataset": spec.dataset,
        "loader": spec.loader,
        "model": spec.model,
        "initializer": spec.initializer,
        "optimizer": spec.optimizer,
        "scheduler": spec.scheduler,
        "loss": spec.loss,
        "training": spec.training,
        "evaluation": spec.evaluation,
        "numerics": spec.numerics,
        "checkpoint": spec.checkpoint,
        "profiling": spec.profiling,
    }


def build_identity(
    spec: RunSpec,
    condition_config: dict[str, object],
    seeds: dict[str, int],
) -> RunIdentity:
    return RunIdentity(
        schema_version=SCHEMA_VERSION,
        project_name="mlprosection",
        experiment_ids=spec.experiment_ids,
        atomic_run_id=spec.atomic_run_id,
        execution_group_id=spec.execution_group_id,
        recipe_id=spec.recipe_id,
        structure_signature=spec.structure_signature,
        condition_key=make_condition_key(condition_config),
        run_key=make_run_key(condition_config, seeds),
        master_seed=seeds["master"],
    )


def build_tags(
    identity: RunIdentity,
    spec: RunSpec,
    git_info: dict[str, object],
    model: Layer | None,
) -> dict[str, str]:
    backend = model.backend if model is not None else get_default_backend()
    return {
        "schema.version": str(identity.schema_version),
        "project.name": identity.project_name,
        "run.type": "seed_trial",
        "code.git_commit": str(git_info["commit"]),
        "code.git_branch": str(git_info["branch"]),
        "code.git_dirty": str(git_info["dirty"]).lower(),
        "code.repository": str(git_info["repository"]),
        "code.entrypoint": str(git_info["entrypoint"]),
        "code.runner_version": "1",
        "runtime.backend": backend.name,
        "runtime.device_type": "cuda" if backend.is_gpu else "cpu",
        "runtime.platform": os.uname().sysname.lower(),
        "runtime.python_version": sys.version.split()[0],
        "atomic_run.id": identity.atomic_run_id,
        "execution_group.id": identity.execution_group_id,
        "recipe.id": identity.recipe_id,
        "structure.signature": identity.structure_signature,
        "condition.key": identity.condition_key,
        "run.key": identity.run_key,
        "master_seed": str(identity.master_seed),
        "dataset.id": str(spec.dataset.get("id", "")),
        "model.family": str(spec.model.get("family", "")),
        "task.type": str(spec.model.get("task_type", "classification")),
        "trial.status": "running",
        "trial.attempt": os.getenv("MLFLOW_TRIAL_ATTEMPT", "1"),
        "retry.of": os.getenv("MLFLOW_RETRY_OF", ""),
        "parent.mlflow_run_id": os.getenv("MLFLOW_PARENT_RUN_ID", ""),
    }


def write_trial_artifacts(
    *,
    artifact_root: Path,
    identity: RunIdentity,
    spec: RunSpec,
    condition_config: dict[str, object],
    seed_config_value: dict[str, int],
    git_info: dict[str, object],
    model: Layer | None,
    final_metrics: dict[str, float],
    history_rows: list[tuple[str, int, str, float]],
    profiling_metrics: dict[str, int | float],
    checkpoint_path: Path | None,
    checkpoint_digest: str | None,
) -> None:
    resolved = {
        **condition_config,
        "condition_key": identity.condition_key,
        "run_key": identity.run_key,
        "seed": seed_config_value,
        "policy": spec.policy or {},
        "regularization": spec.regularization or {},
    }
    write_json(artifact_root / "config/resolved.json", resolved)
    write_json(artifact_root / "config/condition.json", condition_config)
    write_json(artifact_root / "config/seed.json", seed_config_value)
    write_json(artifact_root / "config/profiling.json", spec.profiling)
    write_json(artifact_root / "code/git.json", git_info)
    if git_info["dirty"]:
        write_git_diff(artifact_root / "code/git.diff.patch")
    write_text(artifact_root / "environment/python.txt", sys.version)
    write_text(artifact_root / "environment/packages.txt", pip_freeze())
    write_json(artifact_root / "environment/system.json", environment_artifacts())
    backend = model.backend if model is not None else get_default_backend()
    write_json(
        artifact_root / "environment/backend.json",
        {"backend": backend.name, "device": backend.device, "dtype": backend.dtype_name},
    )
    write_json(artifact_root / "environment/device.json", backend.memory_info())
    write_json(artifact_root / "data/dataset_manifest.json", spec.dataset)
    write_json(artifact_root / "model/architecture.json", spec.model)
    write_text(artifact_root / "model/structure.txt", str(model) if model is not None else "")
    write_json(
        artifact_root / "model/parameter_manifest.json",
        parameter_manifest(model) if model is not None else [],
    )
    write_json(artifact_root / "model/initialization_manifest.json", spec.initializer)
    write_history_csv(artifact_root / "metrics/history.csv", run_key=identity.run_key, rows=history_rows)
    write_history_csv(artifact_root / "metrics/epoch_history.csv", run_key=identity.run_key, rows=history_rows)
    write_runtime_history_csv(artifact_root / "metrics/runtime_history.csv", build_runtime_history_rows(profiling_metrics))
    write_memory_history_csv(artifact_root / "metrics/memory_history.csv", build_memory_history_rows(profiling_metrics))
    write_json(artifact_root / "metrics/final.json", final_metrics)
    write_json(
        artifact_root / "profiles/profiling_summary.json",
        {"schema_version": SCHEMA_VERSION, "enabled": spec.profiling.get("enabled", False), "metrics": profiling_metrics},
    )
    write_json(
        artifact_root / "checkpoints/checkpoint_manifest.json",
        {
            "format": "npz" if checkpoint_path is not None else "none",
            "final": None if checkpoint_path is None else {
                "path": str(checkpoint_path),
                "epoch": spec.training.get("max_epochs"),
                "update": final_metrics.get("final/system/total_updates"),
                "digest": checkpoint_digest,
            },
            "best": None,
            "contains": {
                "model": checkpoint_path is not None,
                "optimizer": False,
                "scheduler": False,
                "rng_state": False,
                "training_state": False,
            },
        },
    )


def add_classification_summaries(final_metrics: dict[str, float], trainer: ForwardTrainer) -> None:
    if trainer.losses.train:
        final_metrics["summary/train/loss/normalized_auc"] = normalized_auc(trainer.losses.train)
        final_metrics["summary/train/loss/minimum"] = float(min(trainer.losses.train))
        final_metrics["summary/train/loss/minimum_step"] = float(trainer.losses.train.index(min(trainer.losses.train)))
    if trainer.accuracies.valid:
        final_metrics["summary/valid/accuracy/final"] = float(trainer.accuracies.valid[-1])
        final_metrics["summary/valid/accuracy/maximum"] = float(max(trainer.accuracies.valid))
        final_metrics["summary/valid/accuracy/maximum_epoch"] = float(trainer.accuracies.valid.index(max(trainer.accuracies.valid)))


def add_inference_metrics(final_metrics: dict[str, float], model: Layer, x_test: Tensor) -> None:
    if len(x_test) == 0:
        return

    sample_count = min(len(x_test), env_int("MLPROSECTION_INFERENCE_LIMIT", 256))
    batch = x_test[:sample_count]
    if hasattr(model, "train_flg"):
        model.train_flg = False
    durations = []
    for _ in range(3):
        start = time.perf_counter()
        model.forward(batch)
        durations.append(time.perf_counter() - start)
    values_ms = [value * 1_000 for value in durations]
    final_metrics["final/runtime/inference_images_per_s"] = sample_count / (sum(durations) / len(durations))
    final_metrics["final/runtime/inference_latency_mean_ms"] = sum(values_ms) / len(values_ms)
    final_metrics["final/runtime/inference_latency_median_ms"] = sorted(values_ms)[len(values_ms) // 2]
    final_metrics["final/runtime/inference_latency_p95_ms"] = max(values_ms)
    final_metrics["final/runtime/inference_latency_std_ms"] = float(math.sqrt(sum((x - final_metrics["final/runtime/inference_latency_mean_ms"]) ** 2 for x in values_ms) / len(values_ms)))


def basic_profiling_metrics(start: float, end: float) -> dict[str, float]:
    try:
        import psutil

        memory = float(psutil.Process().memory_info().rss)
    except Exception:
        memory = 0.0

    duration_ms = (end - start) * 1_000
    return {
        "runtime.train_total.mean_ms": duration_ms,
        "memory.run.start.cpu.rss_bytes": memory,
        "memory.train.start.cpu.rss_bytes": memory,
        "memory.train.end.cpu.rss_bytes": memory,
        "memory.run.end.cpu.rss_bytes": memory,
        "memory.peak_sampled.cpu.rss_bytes": memory,
        "profiling.enabled": 0,
        "profiling.profiled_step_count": 0,
    }


def normalized_auc(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(sum(values) / len(values))


def last_or_none(values: list[float]) -> float | None:
    return None if not values else float(values[-1])


def classification_dataset_config(
    *,
    dataset_id: str,
    train_size: int,
    test_size: int,
    flatten: bool,
) -> dict[str, object]:
    return {
        "id": dataset_id,
        "name": "MNIST",
        "version": "official",
        "source": "ossci-datasets.s3.amazonaws.com/mnist",
        "train_size": train_size,
        "test_size": test_size,
        "input_shape": [784] if flatten else [1, 28, 28],
        "target_shape": [],
        "num_classes": 10,
        "normalization": "pixel_0_1",
        "flatten": flatten,
        "channel_order": "flat" if flatten else "NCHW",
    }


def loader_config(batch_size: int, train_size: int, *, shuffle: bool = True) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "sampling_method": "permutation_per_epoch",
        "num_workers": 0,
        "steps_per_epoch": math.ceil(train_size / batch_size),
        "samples_per_epoch": train_size,
    }


def common_policy(seed_count: int = 10) -> dict[str, object]:
    return {
        "seed_count": seed_count,
        "seed_start": 0,
        "paired_execution": True,
        "save_final_checkpoint": True,
        "save_best_checkpoint": False,
        "fail_on_nan": True,
        "fail_on_inf": True,
        "fail_on_missing_metric": True,
        "resume_allowed": True,
        "retry_allowed": True,
    }


def checkpoint_config() -> dict[str, object]:
    return {
        "save_final": True,
        "save_best": False,
        "format": "npz",
        "include_optimizer": False,
        "include_scheduler": False,
        "include_rng_state": False,
        "include_training_state": False,
    }


def numerics_config(gpu: bool) -> dict[str, object]:
    backend = "cupy" if gpu else "numpy"
    return {
        "dtype": "float32",
        "compute_dtype": "float32",
        "parameter_dtype": "float32",
        "backend": backend,
        "device": "cuda" if gpu else "cpu",
        "deterministic": False,
        "nan_policy": "fail",
        "inf_policy": "fail",
        "epsilon": 1e-7,
    }
