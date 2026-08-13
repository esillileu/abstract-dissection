"""Final-checkpoint full-train accuracy and train/test gap metrics."""

from __future__ import annotations

from pathlib import Path

from mlprosection.experiment.checkpoint import load_model_checkpoint

from ..result_schema import TRAIN_FULL_ACCURACY, TRAIN_TEST_ACCURACY_GAP

TARGET_RUNS = {
    ("GT06", "CNN-SIMPLE-BOOK"),
    ("GT07", "CNN-DEEP-BOOK"),
    ("GT09", "MLP-EXT-ALL-BOOK"),
}


def is_target_run(config: dict[str, object]) -> bool:
    return (
        str(config.get("execution_group_id", "")),
        str(config.get("atomic_run_id", "")),
    ) in TARGET_RUNS


def evaluate_checkpoint_gap(
    *,
    trainer,
    model,
    checkpoint: str | Path,
    x_train,
    t_train,
    x_test,
    t_test,
) -> tuple[object, object, dict[str, float]]:
    """Reload a final checkpoint and evaluate full train and test datasets."""
    load_model_checkpoint(checkpoint, model)
    train = trainer.evaluate(x_train, t_train)
    test = trainer.evaluate(x_test, t_test)
    if train.accuracy is None or test.accuracy is None:
        raise ValueError("full train/test accuracy evaluation returned no accuracy")
    train_accuracy = float(train.accuracy)
    test_accuracy = float(test.accuracy)
    return train, test, {
        TRAIN_FULL_ACCURACY: train_accuracy,
        TRAIN_TEST_ACCURACY_GAP: train_accuracy - test_accuracy,
    }
