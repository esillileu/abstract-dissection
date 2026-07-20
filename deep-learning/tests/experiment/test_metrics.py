from mlprosection.experiment.metrics import evaluation_history, update_history


def test_update_history_uses_monotonic_global_steps() -> None:
    rows = update_history(train_logs=[
        {"epoch": 1, "iteration": 20, "global_step": 20, "loss": 1.5, "elapsed_time": 0.1},
        {"epoch": 2, "iteration": 20, "global_step": 40, "loss": 0.5, "elapsed_time": 0.2},
    ])

    assert rows == [
        ("update", 20, "train/loss", 1.5),
        ("update", 40, "train/loss", 0.5),
    ]


def test_evaluation_history_uses_a_distinct_evaluation_axis() -> None:
    rows = evaluation_history(valid_logs=[
        {"epoch": 1, "iteration": 20, "eval_step": 1, "loss": 1.2, "elapsed_time": 0.1},
        {"epoch": 2, "iteration": 20, "eval_step": 2, "loss": 0.8, "elapsed_time": 0.2},
    ])

    assert rows == [
        ("eval", 1, "valid/loss", 1.2),
        ("eval", 2, "valid/loss", 0.8),
    ]
