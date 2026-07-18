from __future__ import annotations

from common import (
    RunSpec,
    checkpoint_config,
    classification_dataset_config,
    common_policy,
    env_flag,
    env_int,
    load_mnist_tensors,
    loader_config,
    make_deep_cnn,
    make_simple_cnn,
    numerics_config,
    parser,
    profiling_config_dict,
    profiling_config_from_env,
    run_classification_trial,
)


ATOMIC_RUNS = {
    "CNN-SIMPLE": {
        "factory": make_simple_cnn,
        "group": "g08",
        "signature": "mnist-simpleconvnet-v1",
        "name": "SimpleConvNet",
        "conv_layers": 1,
        "dropout": False,
        "flops": 8_726_000,
        "macs": 4_363_000,
    },
    "CNN-DEEP": {
        "factory": make_deep_cnn,
        "group": "g09",
        "signature": "mnist-deepconvnet-v1",
        "name": "DeepConvNet",
        "conv_layers": 6,
        "dropout": True,
        "flops": 11_800_000,
        "macs": 5_900_000,
    },
}


def build_spec(atomic_run_id: str, *, train_size: int, test_size: int, batch_size: int, max_epoch: int, profiling: dict[str, object]) -> RunSpec:
    config = ATOMIC_RUNS[atomic_run_id]
    return RunSpec(
        atomic_run_id=atomic_run_id,
        experiment_ids=("e08",),
        execution_group_id=str(config["group"]),
        recipe_id="RC-CNN",
        structure_signature=str(config["signature"]),
        dataset=classification_dataset_config(dataset_id="DS-MNIST-IMG", train_size=train_size, test_size=test_size, flatten=False),
        loader=loader_config(batch_size, train_size),
        model={"name": config["name"], "family": "cnn", "task_type": "classification", "input_shape": [1, 28, 28], "output_shape": [10], "num_conv_layers": config["conv_layers"], "activation": "relu", "normalization": "none", "use_batchnorm": False, "use_dropout": config["dropout"], "structure_signature": config["signature"], "model/flops": config["flops"], "model/macs": config["macs"]},
        initializer={"name": "he", "fan_mode": "fan_in", "seed": "seed/model_init"},
        optimizer={"name": "adam", "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-7},
        scheduler={"name": "constant"},
        loss={"name": "SoftmaxWithLoss", "reduction": "mean"},
        training={"max_epochs": max_epoch, "batch_size": batch_size, "log_interval": env_int("MLPROSECTION_LOG_INTERVAL", 10), "entrypoint": "experiments/run/e08_cnn_structure.py"},
        evaluation={"batch_size": batch_size, "use_full_train": True, "use_full_test": True, "primary_metric": "test/accuracy", "checkpoint_selection": "final"},
        numerics=numerics_config(env_flag("MLPROSECTION_GPU", "0")),
        checkpoint=checkpoint_config(),
        profiling=profiling,
        policy=common_policy(),
    )


def main() -> None:
    args = parser("Run e08 CNN structure atomic trial.", sorted(ATOMIC_RUNS)).parse_args()
    gpu = env_flag("MLPROSECTION_GPU", "0")
    train_limit = env_int("MLPROSECTION_TRAIN_LIMIT", 60000)
    test_limit = env_int("MLPROSECTION_TEST_LIMIT", 10000)
    batch_size = env_int("MLPROSECTION_BATCH_SIZE", 100)
    max_epoch = env_int("MLPROSECTION_MAX_EPOCHS", 20)
    (x_train, t_train), (x_test, t_test) = load_mnist_tensors(flatten=False, gpu=gpu, train_limit=train_limit, test_limit=test_limit)
    config = ATOMIC_RUNS[args.atomic_run_id]
    model = config["factory"]()
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
        optimizer_name="adam",
        learning_rate=0.001,
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
