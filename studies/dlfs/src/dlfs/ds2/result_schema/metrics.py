"""DS2 normalized metric declarations."""

from dlfs.analysis.declarations import MetricDeclaration

LOSS = MetricDeclaration(
    "train_loss",
    "nats",
    "train",
    "run",
    ("final/train/loss", "train/loss"),
    ("final/train/loss", "train/loss"),
    protocols=("book-source-v1", "legacy"),
)
TRAIN_PPL = MetricDeclaration(
    "train_perplexity",
    "perplexity",
    "train",
    "run",
    ("final/train/perplexity", "train/perplexity"),
    ("final/train/perplexity", "train/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
TEST_PPL = MetricDeclaration(
    "test_perplexity",
    "perplexity",
    "test",
    "run",
    ("final/test/perplexity", "test/perplexity"),
    ("final/test/perplexity", "test/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
EXACT = MetricDeclaration(
    "test_exact_match",
    "percent",
    "test",
    "run",
    ("final/test/exact_match", "test/exact_match"),
    ("final/test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)
SUMMARY_TRAIN_LOSS = MetricDeclaration(
    "train_loss",
    "nats",
    "train",
    "run",
    ("final/train/loss", "train/loss"),
    ("final/train/loss", "train/loss"),
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
SUMMARY_BOOK_LOSS = MetricDeclaration(
    "book_loss",
    "nats",
    "train",
    "run",
    (
        "final/train/book_loss",
        "update/train/book_loss",
        "series/train/book_loss",
    ),
    ("final/train/loss", "train/loss"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TRAIN_PPL = MetricDeclaration(
    "train_perplexity",
    "perplexity",
    "train",
    "run",
    ("final/train/perplexity", "train/perplexity"),
    ("final/train/perplexity", "train/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TEST_PPL = MetricDeclaration(
    "test_perplexity",
    "perplexity",
    "test",
    "run",
    ("final/test/perplexity", "test/perplexity"),
    ("final/test/perplexity", "test/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_VALID_PPL = MetricDeclaration(
    "validation_perplexity",
    "perplexity",
    "validation",
    "run",
    ("final/valid/perplexity", "valid/perplexity"),
    ("final/valid/perplexity", "valid/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TRAIN_EXACT = MetricDeclaration(
    "train_exact_match",
    "percent",
    "train",
    "run",
    ("final/train/exact_match", "final/train/accuracy", "train/accuracy"),
    ("final/train/exact_match", "final/train/accuracy", "train/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)
SUMMARY_TEST_EXACT = MetricDeclaration(
    "test_exact_match",
    "percent",
    "test",
    "run",
    ("final/test/exact_match", "final/test/accuracy", "test/accuracy"),
    ("final/test/exact_match", "final/test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)
SUMMARY_TEST_ACCURACY = MetricDeclaration(
    "test_accuracy",
    "percent",
    "test",
    "run",
    ("final/test/exact_match", "final/test/accuracy", "test/accuracy"),
    ("final/test/exact_match", "final/test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)
PROFILE_POINTS_OK = MetricDeclaration(
    "profile_points_ok",
    "count",
    "profile",
    "run",
    ("profile/points/ok",),
    (),
    protocols=(
        "ds2-e10-profile-v1",
        "ds2-e11-vocabulary-size-scaling-v1",
    ),
)
COUNT_PIPELINE_TIME = MetricDeclaration(
    "count_pipeline_time",
    "seconds",
    "run",
    "run",
    ("runtime/train_total_s",),
    ("runtime/train_total_s",),
    protocols=("ds2-e12-count-based-v1", "legacy"),
)

__all__ = [
    "COUNT_PIPELINE_TIME",
    "EXACT",
    "LOSS",
    "PROFILE_POINTS_OK",
    "SUMMARY_BOOK_LOSS",
    "SUMMARY_TEST_ACCURACY",
    "SUMMARY_TEST_EXACT",
    "SUMMARY_TEST_LOSS",
    "SUMMARY_TEST_PPL",
    "SUMMARY_TRAIN_EXACT",
    "SUMMARY_TRAIN_LOSS",
    "SUMMARY_TRAIN_PPL",
    "SUMMARY_VALID_PPL",
    "TEST_PPL",
    "TRAIN_PPL",
]
