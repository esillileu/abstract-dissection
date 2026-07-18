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
    make_mlp,
    numerics_config,
    parser,
    profiling_config_dict,
    profiling_config_from_env,
    run_classification_trial,
)


ATOMIC_RUNS = {
    "REG-DO-01": {"dropout_ratio": 0.1},
    "REG-DO-02": {"dropout_ratio": 0.2},
    "REG-DO-03": {"dropout_ratio": 0.3},
    "REG-DO-05": {"dropout_ratio": 0.5},
}


def build_spec(atomic_run_id: str, *, train_size: int, test_size: int, batch_size: int, max_epoch: int, profiling: dict[str, object]) -> RunSpec:
    dropout_ratio = ATOMIC_RUNS[atomic_run_id]["dropout_ratio"]
    return RunSpec(
        atomic_run_id=atomic_run_id,
        experiment_ids=("e07",),
        execution_group_id="g07",
        recipe_id="RC-REG-DROPOUT",
        structure_signature="mnist-mlp-784-100x6-10-relu-dropout-v1",
        dataset=classification_dataset_config(dataset_id="DS-MNIST-300", train_size=train_size, test_size=test_size, flatten=True),
        loader=loader_config(batch_size, train_size),
        model={"name": "MLP6HiddenDropout", "family": "mlp", "task_type": "classification", "input_shape": [784], "output_shape": [10], "hidden_sizes": [100] * 6, "num_hidden_layers": 6, "activation": "relu", "use_batchnorm": False, "use_dropout": True, "dropout_ratio": dropout_ratio, "structure_signature": "mnist-mlp-784-100x6-10-relu-dropout-v1"},
        initializer={"name": "he", "seed": "seed/model_init"},
        optimizer={"name": "sgd", "learning_rate": 0.01, "weight_decay": 0.0},
        scheduler={"name": "constant"},
        loss={"name": "SoftmaxWithLoss", "reduction": "mean"},
        training={"max_epochs": max_epoch, "batch_size": batch_size, "log_interval": env_int("MLPROSECTION_LOG_INTERVAL", 10), "entrypoint": "experiments/run/e07_dropout.py"},
        evaluation={"batch_size": batch_size, "use_full_train": True, "use_full_test": True, "primary_metric": "test/accuracy", "checkpoint_selection": "final"},
        numerics=numerics_config(env_flag("MLPROSECTION_GPU", "0")),
        checkpoint=checkpoint_config(),
        profiling=profiling,
        regularization={"l2_lambda": 0.0, "dropout_ratio": dropout_ratio, "dropout_mode": "inverted", "dropout_locations": "after_hidden_relu"},
        policy=common_policy(),
    )


def main() -> None:
    args = parser("Run e07 dropout atomic trial.", sorted(ATOMIC_RUNS)).parse_args()
    gpu = env_flag("MLPROSECTION_GPU", "0")
    train_limit = env_int("MLPROSECTION_TRAIN_LIMIT", 300)
    test_limit = env_int("MLPROSECTION_TEST_LIMIT", 10000)
    batch_size = env_int("MLPROSECTION_BATCH_SIZE", 100)
    max_epoch = env_int("MLPROSECTION_MAX_EPOCHS", 301)
    (x_train, t_train), (x_test, t_test) = load_mnist_tensors(flatten=True, gpu=gpu, train_limit=train_limit, test_limit=test_limit)
    dropout_ratio = float(ATOMIC_RUNS[args.atomic_run_id]["dropout_ratio"])
    model = make_mlp(input_size=784, hidden_size=100, hidden_layers=6, output_size=10, initializer="he", dropout_ratio=dropout_ratio)
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
