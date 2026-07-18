from __future__ import annotations

import numpy as np

from common import (
    RunSpec,
    checkpoint_config,
    classification_dataset_config,
    common_policy,
    env_flag,
    env_int,
    load_mnist_tensors,
    loader_config,
    make_mlp,
    numerics_config,
    parser,
    profiling_config_dict,
    profiling_config_from_env,
    run_classification_trial,
)


SCALES = [float(value) for value in np.logspace(0, -4, 16)]
ATOMIC_RUNS = {
    **{f"BN-OFF-{index:02d}": {"batchnorm": False, "scale": scale, "scale_index": index} for index, scale in enumerate(SCALES, 1)},
    **{f"BN-ON-{index:02d}": {"batchnorm": True, "scale": scale, "scale_index": index} for index, scale in enumerate(SCALES, 1)},
}


def build_spec(atomic_run_id: str, *, train_size: int, test_size: int, batch_size: int, max_epoch: int, profiling: dict[str, object]) -> RunSpec:
    config = ATOMIC_RUNS[atomic_run_id]
    group = "g05" if config["batchnorm"] else "g04"
    signature = "mnist-mlp-784-100x5-10-bn-relu-v1" if config["batchnorm"] else "mnist-mlp-784-100x5-10-relu-no-bn-v1"
    return RunSpec(
        atomic_run_id=atomic_run_id,
        experiment_ids=("e05",),
        execution_group_id=group,
        recipe_id="RC-BN-SCALE",
        structure_signature=signature,
        dataset=classification_dataset_config(dataset_id="DS-MNIST-1000", train_size=train_size, test_size=test_size, flatten=True),
        loader=loader_config(batch_size, train_size),
        model={"name": "MLP5Hidden", "family": "mlp", "task_type": "classification", "input_shape": [784], "output_shape": [10], "hidden_sizes": [100] * 5, "num_hidden_layers": 5, "activation": "relu", "use_batchnorm": config["batchnorm"], "use_dropout": False, "structure_signature": signature},
        initializer={"name": "normal", "scale": config["scale"], "scale_index": config["scale_index"], "seed": "seed/model_init"},
        optimizer={"name": "sgd", "learning_rate": 0.01},
        scheduler={"name": "constant"},
        loss={"name": "SoftmaxWithLoss", "reduction": "mean"},
        training={"max_epochs": max_epoch, "batch_size": batch_size, "log_interval": env_int("MLPROSECTION_LOG_INTERVAL", 10), "entrypoint": "experiments/run/e05_batchnorm_scale.py"},
        evaluation={"batch_size": batch_size, "use_full_train": True, "use_full_test": True, "primary_metric": "test/accuracy", "checkpoint_selection": "final"},
        numerics=numerics_config(env_flag("MLPROSECTION_GPU", "0")),
        checkpoint=checkpoint_config(),
        profiling=profiling,
        policy=common_policy(),
    )


def main() -> None:
    args = parser("Run e05 BatchNorm scale atomic trial.", sorted(ATOMIC_RUNS)).parse_args()
    gpu = env_flag("MLPROSECTION_GPU", "0")
    train_limit = env_int("MLPROSECTION_TRAIN_LIMIT", 1000)
    test_limit = env_int("MLPROSECTION_TEST_LIMIT", 10000)
    batch_size = env_int("MLPROSECTION_BATCH_SIZE", 100)
    max_epoch = env_int("MLPROSECTION_MAX_EPOCHS", 20)
    (x_train, t_train), (x_test, t_test) = load_mnist_tensors(flatten=True, gpu=gpu, train_limit=train_limit, test_limit=test_limit)
    config = ATOMIC_RUNS[args.atomic_run_id]
    model = make_mlp(input_size=784, hidden_size=100, hidden_layers=5, output_size=10, initializer=f"std:{config['scale']}", batchnorm=bool(config["batchnorm"]))
    if gpu:
        model.gpu()
    spec = build_spec(args.atomic_run_id, train_size=len(x_train), test_size=len(x_test), batch_size=batch_size, max_epoch=max_epoch, profiling=profiling_config_dict(profiling_config_from_env()))
    run_key = run_classification_trial(
        spec=spec,
        seed=args.seed,
        model=model,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        optimizer_name="sgd",
        learning_rate=0.01,
        weight_decay=0.0,
        max_epoch=max_epoch,
        batch_size=batch_size,
        log_interval=env_int("MLPROSECTION_LOG_INTERVAL", 10),
        tracking_uri=args.tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_enabled=not args.no_mlflow,
    )
    print(f"run_key={run_key}")


if __name__ == "__main__":
    main()
