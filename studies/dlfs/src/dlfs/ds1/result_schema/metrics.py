"""DS1 normalized metric and summary declarations."""

from dlfs.analysis.declarations import MetricDeclaration

TRAIN_FULL_ACCURACY = "final/train-full/accuracy"
TRAIN_TEST_ACCURACY_GAP = "final/train-test/accuracy-gap"

ACCURACY = MetricDeclaration(
    "test_accuracy",
    "fraction",
    "test",
    "run",
    ("final/test/accuracy", "test/accuracy"),
    ("final/test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
TRAIN_LOSS = MetricDeclaration(
    "train_loss",
    "nats",
    "train",
    "update",
    ("train/loss", "train/objective"),
    ("train/loss", "train/objective"),
    protocols=("book-source-v1", "legacy"),
)
TRAIN_ACCURACY_CURVE = MetricDeclaration(
    "train_accuracy_curve",
    "fraction",
    "train",
    "epoch",
    ("update/eval_train/accuracy", "train/accuracy"),
    ("update/eval_train/accuracy", "train/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
TEST_ACCURACY_CURVE = MetricDeclaration(
    "test_accuracy_curve",
    "fraction",
    "test",
    "epoch",
    ("update/eval_test/accuracy", "test/accuracy"),
    ("update/eval_test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TRAIN_ACCURACY = MetricDeclaration(
    "train_accuracy",
    "fraction",
    "train",
    "run",
    ("final/train-full/accuracy", "final/train/accuracy", "train/accuracy"),
    ("final/train-full/accuracy", "final/train/accuracy", "train/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TEST_ACCURACY = MetricDeclaration(
    "test_accuracy",
    "fraction",
    "test",
    "run",
    ("final/test/accuracy", "final/test-full/accuracy", "test/accuracy"),
    ("final/test/accuracy", "final/test-full/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TRAIN_ACCURACY_PERCENT = MetricDeclaration(
    "train_accuracy",
    "percent",
    "train",
    "run",
    ("final/train-full/accuracy", "final/train/accuracy", "train/accuracy"),
    ("final/train-full/accuracy", "final/train/accuracy", "train/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)
SUMMARY_TEST_ACCURACY_PERCENT = MetricDeclaration(
    "test_accuracy",
    "percent",
    "test",
    "run",
    ("final/test/accuracy", "final/test-full/accuracy", "test/accuracy"),
    ("final/test/accuracy", "final/test-full/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)
SUMMARY_TRAIN_LOSS = MetricDeclaration(
    "train_loss",
    "nats",
    "train",
    "run",
    ("final/train/loss", "train/loss", "train/objective"),
    ("final/train/loss", "train/loss", "train/objective"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TEST_LOSS = MetricDeclaration(
    "test_loss",
    "nats",
    "test",
    "run",
    ("final/test/loss", "test/loss"),
    ("final/test/loss", "test/loss"),
    protocols=("book-source-v1", "legacy"),
)
GRADIENT_CHECK_SUMMARIES = (
    *tuple(
        MetricDeclaration(
            f"{parameter.lower()}_mean_absolute_difference",
            "absolute_gradient",
            "gradient_check",
            "run",
            (
                f"gradient_check/{parameter}/mean_absolute_difference",
                f"observation/gradient_check/{parameter}/mean_absolute_difference",
            ),
            (f"observation/gradient_check/{parameter}/mean_absolute_difference",),
            protocols=("book-source-v1", "legacy"),
        )
        for parameter in ("W1", "b1", "W2", "b2")
    ),
    MetricDeclaration(
        "numerical_gradient_time",
        "seconds",
        "gradient_check",
        "run",
        ("gradient_check/numerical_s", "observation/gradient_check/numerical_s"),
        ("observation/gradient_check/numerical_s",),
        protocols=("book-source-v1", "legacy"),
    ),
    MetricDeclaration(
        "backprop_gradient_time",
        "seconds",
        "gradient_check",
        "run",
        ("gradient_check/backprop_s", "observation/gradient_check/backprop_s"),
        ("observation/gradient_check/backprop_s",),
        protocols=("book-source-v1", "legacy"),
    ),
    MetricDeclaration(
        "gradient_time_speedup",
        "ratio",
        "gradient_check",
        "run",
        ("gradient_check/speedup", "observation/gradient_check/speedup"),
        ("observation/gradient_check/speedup",),
        protocols=("book-source-v1", "legacy"),
    ),
)
