from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from mlprosection.core.backend import get_default_backend
from mlprosection.datasets import load_mnist
from mlprosection.nn.initailizer import he_normal_
from mlprosection.nn.layers.criterion import SoftmaxWithLoss
from mlprosection.nn.model.test import SimpleCNN
from mlprosection.optim.SGD import SGD
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


LOG_PATH = Path("experiments/logs/00_two_layer_net.log")
PARAM_PATH = Path("experiments/params/cnn.npz")
ARTIFACT_ROOT = Path("experiments/results/mlflow_artifacts")
SCHEMA_VERSION = 1


def _configure_logger(*, mode: str = "w") -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("experiments.00_two_layer_net")
    logger.setLevel(logging.INFO)
    logger.disabled = False
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, mode=mode, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _limit_dataset(x, t, env_name: str):
    limit = os.getenv(env_name)
    if limit is None:
        return x, t

    size = int(limit)
    return x[:size], t[:size]


def _last_or_none(values):
    if not values:
        return None
    return values[-1]


def _log_metric(logger: logging.Logger, name: str, value) -> None:
    if value is not None:
        logger.info("%s=%s", name, value)


def _enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _seed_config(master_seed: int) -> dict[str, int]:
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


def _profiling_config_dict(config: ProfilingConfig) -> dict[str, object]:
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


def _build_condition_config(
    *,
    git_info: dict[str, object],
    train_size: int,
    test_size: int,
    max_epoch: int,
    batch_size: int,
    log_interval: int,
    profiling_config: ProfilingConfig,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "atomic_run_id": "CNN-SGD-HE",
        "execution_group_id": "g-cnn-smoke",
        "recipe_id": "RC-CNN",
        "structure_signature": "mnist-cnn-simple-1x28x28-conv30-fc100-10-relu-v1",
        "code": {
            "git_commit": git_info["commit"],
            "entrypoint": "experiments/00_two_layer_net.py",
        },
        "dataset": {
            "id": "DS-MNIST-IMG",
            "name": "MNIST",
            "version": "official",
            "source": "ossci-datasets.s3.amazonaws.com/mnist",
            "train_size": train_size,
            "test_size": test_size,
            "input_shape": [1, 28, 28],
            "target_shape": [],
            "num_classes": 10,
            "normalization": "pixel_0_1",
            "flatten": False,
            "channel_order": "NCHW",
        },
        "loader": {
            "batch_size": batch_size,
            "shuffle": True,
            "drop_last": False,
            "sampling_method": "permutation_per_epoch",
            "num_workers": 0,
            "steps_per_epoch": (train_size + batch_size - 1) // batch_size,
            "samples_per_epoch": train_size,
        },
        "model": {
            "name": "SimpleCNN",
            "family": "cnn",
            "version": "v1",
            "input_shape": [1, 28, 28],
            "output_shape": [10],
            "hidden_size": 100,
            "num_conv_layers": 1,
            "activation": "relu",
            "normalization": "none",
            "use_batchnorm": False,
            "use_dropout": False,
            "structure_signature": "mnist-cnn-simple-1x28x28-conv30-fc100-10-relu-v1",
        },
        "initializer": {
            "name": "he_normal",
            "fan_mode": "fan_in",
            "seed": "seed/model_init",
        },
        "optimizer": {
            "name": "SGD",
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "weight_decay_mode": "coupled_l2_gradient",
        },
        "scheduler": {"name": "constant"},
        "loss": {"name": "SoftmaxWithLoss", "reduction": "mean"},
        "training": {
            "max_epochs": max_epoch,
            "batch_size": batch_size,
            "max_grad": None,
            "gradient_clip_type": "none",
            "update_rule": "sgd",
            "log_interval": log_interval,
            "log_interval_unit": "update",
        },
        "evaluation": {
            "batch_size": batch_size,
            "use_full_train": True,
            "use_full_test": True,
            "primary_metric": "test/accuracy",
            "checkpoint_selection": "final",
        },
        "numerics": {
            "dtype": "float32",
            "compute_dtype": "float32",
            "parameter_dtype": "float32",
            "backend": "cupy",
            "device": "cuda",
            "deterministic": False,
            "nan_policy": "fail",
            "inf_policy": "fail",
            "epsilon": 1e-7,
        },
        "checkpoint": {
            "save_final": True,
            "save_best": False,
            "format": "npz",
            "include_optimizer": False,
            "include_scheduler": False,
            "include_rng_state": False,
            "include_training_state": False,
        },
        "profiling": _profiling_config_dict(profiling_config),
    }


def _build_identity(
    *,
    condition_config: dict[str, object],
    seed_config: dict[str, int],
) -> RunIdentity:
    condition_key = make_condition_key(condition_config)
    run_key = make_run_key(condition_config, seed_config)
    return RunIdentity(
        schema_version=SCHEMA_VERSION,
        project_name="mlprosection",
        experiment_ids=("e-cnn-smoke",),
        atomic_run_id=str(condition_config["atomic_run_id"]),
        execution_group_id=str(condition_config["execution_group_id"]),
        recipe_id=str(condition_config["recipe_id"]),
        structure_signature=str(condition_config["structure_signature"]),
        condition_key=condition_key,
        run_key=run_key,
        master_seed=seed_config["master"],
    )


def _build_tags(
    *,
    identity: RunIdentity,
    git_info: dict[str, object],
    trainer: ForwardTrainer,
) -> dict[str, str]:
    backend = trainer.backend
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
        "runtime.python_version": os.sys.version.split()[0],
        "atomic_run.id": identity.atomic_run_id,
        "execution_group.id": identity.execution_group_id,
        "recipe.id": identity.recipe_id,
        "structure.signature": identity.structure_signature,
        "condition.key": identity.condition_key,
        "run.key": identity.run_key,
        "master_seed": str(identity.master_seed),
        "dataset.id": "DS-MNIST-IMG",
        "model.family": "cnn",
        "task.type": "classification",
        "trial.status": "running",
        "trial.attempt": os.getenv("MLFLOW_TRIAL_ATTEMPT", "1"),
        "retry.of": os.getenv("MLFLOW_RETRY_OF", ""),
        "parent.mlflow_run_id": os.getenv("MLFLOW_PARENT_RUN_ID", ""),
    }


def _write_artifacts(
    *,
    artifact_root: Path,
    identity: RunIdentity,
    condition_config: dict[str, object],
    seed_config: dict[str, int],
    git_info: dict[str, object],
    initial_parameter_manifest: list[dict[str, object]],
    trainer: ForwardTrainer,
    final_metrics: dict[str, float],
    epoch_rows: list[tuple[str, int, str, float]],
    profiling_metrics: dict[str, int | float],
) -> None:
    resolved_config = {
        **condition_config,
        "condition_key": identity.condition_key,
        "run_key": identity.run_key,
        "seed": seed_config,
    }
    write_json(artifact_root / "config/resolved.json", resolved_config)
    write_json(artifact_root / "config/condition.json", condition_config)
    write_json(artifact_root / "config/seed.json", seed_config)
    write_json(artifact_root / "config/profiling.json", condition_config["profiling"])
    write_json(artifact_root / "code/git.json", git_info)
    if git_info["dirty"]:
        write_git_diff(artifact_root / "code/git.diff.patch")
    write_text(artifact_root / "environment/python.txt", os.sys.version)
    write_text(artifact_root / "environment/packages.txt", pip_freeze())
    write_json(artifact_root / "environment/system.json", environment_artifacts())
    write_json(
        artifact_root / "environment/backend.json",
        {
            "backend": trainer.backend.name,
            "device": trainer.backend.device,
            "dtype": trainer.backend.dtype_name,
        },
    )
    write_json(artifact_root / "environment/device.json", trainer.backend.memory_info())
    write_json(artifact_root / "data/dataset_manifest.json", condition_config["dataset"])
    write_json(artifact_root / "model/architecture.json", condition_config["model"])
    write_text(artifact_root / "model/structure.txt", str(trainer.model))
    write_json(
        artifact_root / "model/parameter_manifest.json",
        initial_parameter_manifest,
    )
    write_json(artifact_root / "model/initialization_manifest.json", {
        "initializer": condition_config["initializer"],
    })
    write_history_csv(
        artifact_root / "metrics/history.csv",
        run_key=identity.run_key,
        rows=epoch_rows,
    )
    write_history_csv(
        artifact_root / "metrics/epoch_history.csv",
        run_key=identity.run_key,
        rows=epoch_rows,
    )
    write_runtime_history_csv(
        artifact_root / "metrics/runtime_history.csv",
        build_runtime_history_rows(profiling_metrics),
    )
    write_memory_history_csv(
        artifact_root / "metrics/memory_history.csv",
        build_memory_history_rows(profiling_metrics),
    )
    write_json(artifact_root / "metrics/final.json", final_metrics)
    write_json(
        artifact_root / "profiles/profiling_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "enabled": trainer.profiling_config.enabled,
            "profiled_update_range": {
                "start": trainer.profiling_config.start_step,
                "end": trainer.profiling_config.start_step
                + trainer.profiling_config.num_steps
                - 1,
            },
            "metrics": profiling_metrics,
        },
    )


def main() -> None:
    logger = _configure_logger()

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, gpu=True)
    x_train, t_train = _limit_dataset(x_train, t_train, "MLPROSECTION_TRAIN_LIMIT")
    x_test, t_test = _limit_dataset(x_test, t_test, "MLPROSECTION_TEST_LIMIT")

    max_epoch = int(os.getenv("MLPROSECTION_EPOCHS", "5"))
    batch_size = int(os.getenv("MLPROSECTION_BATCH_SIZE", "64"))
    log_interval = int(os.getenv("MLPROSECTION_LOG_INTERVAL", "30"))
    master_seed = int(os.getenv("MLPROSECTION_MASTER_SEED", "0"))
    get_default_backend().seed(master_seed)

    logger.info("experiment=00_two_layer_net")
    logger.info("train_size=%s", len(x_train))
    logger.info("test_size=%s", len(x_test))
    logger.info("max_epoch=%s", max_epoch)
    logger.info("batch_size=%s", batch_size)

    model = SimpleCNN().gpu()

    model.layers[4].reset_weight(he_normal_)
    model.layers[6].reset_weight(he_normal_)
    initial_parameter_manifest = parameter_manifest(model)

    initial_prediction = model.forward(x_train[10:20]).argmax(axis=1).data
    logger.info("initial_prediction=%s", initial_prediction)
    logger.info("initial_target=%s", t_train[10:20].data)

    criterion = SoftmaxWithLoss().gpu()
    optimizer = SGD(model.named_parameters(), pre_step_hooks=[L2Regularization()])
    profiling_config = ProfilingConfig(
        enabled=True,
        start_step=0,
        num_steps=5,
        profile_memory=True,
        profile_gpu_ranges=True,
        sample_memory_every_n_steps=20,
    )
    git_info = current_git_info("experiments/00_two_layer_net.py")
    seed_config = _seed_config(master_seed)
    condition_config = _build_condition_config(
        git_info=git_info,
        train_size=len(x_train),
        test_size=len(x_test),
        max_epoch=max_epoch,
        batch_size=batch_size,
        log_interval=log_interval,
        profiling_config=profiling_config,
    )
    identity = _build_identity(
        condition_config=condition_config,
        seed_config=seed_config,
    )

    trainer = ForwardTrainer(
        model,
        criterion,
        optimizer,
        max_epoch,
        batch_size,
        log_interval,
        profiling_config=profiling_config,
    )
    mlflow_logger = None
    if _enabled("MLFLOW_ENABLE", "1"):
        mlflow_logger = MLflowRunLogger(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlprosection"),
        )
        tags = _build_tags(identity=identity, git_info=git_info, trainer=trainer)
        params = flatten_dict({**condition_config, "seed": seed_config})
        mlflow_logger.start_child(
            run_name=f"{identity.atomic_run_id}-s{identity.master_seed:02d}",
            tags=tags,
            params=params,
        )
        logger = _configure_logger(mode="a")

    run_start = time.perf_counter()
    try:
        trainer.fit(x_train[:], t_train[:], x_test[:], t_test[:])
    except BaseException as exc:
        if mlflow_logger is not None:
            mlflow_logger.finalize_failure(exc)
        raise

    final_prediction = model.forward(x_train[10:20]).argmax(axis=1).data
    logger.info("final_prediction=%s", final_prediction)
    logger.info("final_target=%s", t_train[10:20].data)

    _log_metric(logger, "train_loss", _last_or_none(trainer.losses.train))
    _log_metric(logger, "test_loss", _last_or_none(trainer.losses.valid))
    _log_metric(logger, "train_accuracy", _last_or_none(trainer.accuracies.train))
    _log_metric(logger, "test_accuracy", _last_or_none(trainer.accuracies.valid))

    profiling_metrics = trainer.profiling_metrics()
    final_metrics = build_schema_metrics(
        train_loss=_last_or_none(trainer.losses.train),
        test_loss=_last_or_none(trainer.losses.valid),
        train_accuracy=_last_or_none(trainer.accuracies.train),
        test_accuracy=_last_or_none(trainer.accuracies.valid),
        profiling_metrics=profiling_metrics,
        total_updates=trainer.global_step,
        completed_epochs=max_epoch,
        samples_seen=len(x_train) * max_epoch,
    )
    final_metrics["runtime/run_wall_total_s"] = time.perf_counter() - run_start
    epoch_rows = build_epoch_metric_rows(
        train_losses=trainer.losses.train,
        test_losses=trainer.losses.valid,
        train_accuracies=trainer.accuracies.train,
        test_accuracies=trainer.accuracies.valid,
        profiling_metrics=profiling_metrics,
    )

    artifact_root = ARTIFACT_ROOT / identity.run_key
    artifact_start = time.perf_counter()
    PARAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_params_npz(PARAM_PATH)
    final_metrics["runtime/checkpoint_total_s"] = time.perf_counter() - artifact_start
    checkpoint_digest = file_digest(PARAM_PATH)
    write_json(
        artifact_root / "checkpoints/checkpoint_manifest.json",
        {
            "format": "npz",
            "final": {
                "path": str(PARAM_PATH),
                "epoch": max_epoch,
                "update": trainer.global_step,
                "digest": checkpoint_digest,
            },
            "best": None,
            "contains": {
                "model": True,
                "optimizer": False,
                "scheduler": False,
                "rng_state": False,
                "training_state": False,
            },
        },
    )
    _write_artifacts(
        artifact_root=artifact_root,
        identity=identity,
        condition_config=condition_config,
        seed_config=seed_config,
        git_info=git_info,
        initial_parameter_manifest=initial_parameter_manifest,
        trainer=trainer,
        final_metrics=final_metrics,
        epoch_rows=epoch_rows,
        profiling_metrics=profiling_metrics,
    )
    artifact_logging_start = time.perf_counter()
    if mlflow_logger is not None:
        for step_type, step, metric, value in epoch_rows:
            if step_type == "epoch":
                mlflow_logger.log_epoch_metrics(step, {f"epoch/{metric}": value})
        mlflow_logger.log_final_metrics(final_metrics)
        mlflow_logger.log_artifact_tree(artifact_root)
        final_metrics["runtime/artifact_logging_total_s"] = (
            time.perf_counter() - artifact_logging_start
        )
        mlflow_logger.log_final_metrics(
            {
                "runtime/artifact_logging_total_s": final_metrics[
                    "runtime/artifact_logging_total_s"
                ]
            }
        )
        mlflow_logger.finalize_success()
    else:
        final_metrics["runtime/artifact_logging_total_s"] = 0.0
    write_json(artifact_root / "metrics/final.json", final_metrics)

    logger.info("profiling_metrics_begin")
    for name, value in sorted(profiling_metrics.items()):
        logger.info("%s=%s", name, value)
    logger.info("profiling_metrics_end")

    logger.info("mlflow_enabled=%s", mlflow_logger is not None)
    logger.info("mlflow_run_key=%s", identity.run_key)
    logger.info("mlflow_artifacts=%s", artifact_root)
    logger.info("saved_params=%s", PARAM_PATH)
    logger.info("log_path=%s", LOG_PATH)


if __name__ == "__main__":
    main()
