"""Complete DS2 normalized condition and metric declarations."""

from exp.deepscratch.analysis.declarations import (
    ConditionDeclaration,
    MetricDeclaration,
    StudyDeclaration,
)


LOSS = MetricDeclaration(
    "train_loss", "nats", "train", "run",
    ("final/train/loss", "train/loss"), ("final/train/loss", "train/loss"),
    protocols=("book-source-v1", "legacy"),
)
TRAIN_PPL = MetricDeclaration(
    "train_perplexity", "perplexity", "train", "run",
    ("final/train/perplexity", "train/perplexity"),
    ("final/train/perplexity", "train/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
TEST_PPL = MetricDeclaration(
    "test_perplexity", "perplexity", "test", "run",
    ("final/test/perplexity", "test/perplexity"),
    ("final/test/perplexity", "test/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
EXACT = MetricDeclaration(
    "test_exact_match", "percent", "test", "run",
    ("final/test/exact_match", "test/exact_match"),
    ("final/test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)
SUMMARY_TRAIN_LOSS = MetricDeclaration(
    "train_loss", "nats", "train", "run",
    ("final/train/loss", "train/loss"),
    ("final/train/loss", "train/loss"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TEST_LOSS = MetricDeclaration(
    "test_loss", "nats", "test", "run",
    ("final/test/loss", "test/loss"),
    ("final/test/loss", "test/loss"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_BOOK_LOSS = MetricDeclaration(
    "book_loss", "nats", "train", "run",
    (
        "final/train/book_loss",
        "update/train/book_loss",
        "series/train/book_loss",
    ),
    ("final/train/loss", "train/loss"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TRAIN_PPL = MetricDeclaration(
    "train_perplexity", "perplexity", "train", "run",
    ("final/train/perplexity", "train/perplexity"),
    ("final/train/perplexity", "train/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TEST_PPL = MetricDeclaration(
    "test_perplexity", "perplexity", "test", "run",
    ("final/test/perplexity", "test/perplexity"),
    ("final/test/perplexity", "test/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_VALID_PPL = MetricDeclaration(
    "validation_perplexity", "perplexity", "validation", "run",
    ("final/valid/perplexity", "valid/perplexity"),
    ("final/valid/perplexity", "valid/perplexity"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TRAIN_EXACT = MetricDeclaration(
    "train_exact_match", "percent", "train", "run",
    ("final/train/exact_match", "final/train/accuracy", "train/accuracy"),
    ("final/train/exact_match", "final/train/accuracy", "train/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)
SUMMARY_TEST_EXACT = MetricDeclaration(
    "test_exact_match", "percent", "test", "run",
    ("final/test/exact_match", "final/test/accuracy", "test/accuracy"),
    ("final/test/exact_match", "final/test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)
SUMMARY_TEST_ACCURACY = MetricDeclaration(
    "test_accuracy", "percent", "test", "run",
    ("final/test/exact_match", "final/test/accuracy", "test/accuracy"),
    ("final/test/exact_match", "final/test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
    value_scale=100.0,
)


def condition(canonical, implemented, original=(), metric=LOSS):
    return ConditionDeclaration(canonical, tuple(implemented), tuple(original), (metric,))


E01 = StudyDeclaration("e01", (
    condition(
        "toy-cbow",
        ("W2V-TOY-CBOW-FULL",),
        ("TOY-CBOW",),
    ),
    condition(
        "toy-skipgram",
        ("W2V-TOY-SKIPGRAM-FULL",),
        ("TOY-SKIPGRAM",),
    ),
))
E02_NAMES = (
    "W2V-PTB-CBOW-NS", "W2V-PTB-SKIPGRAM-NS", "W2V-PTB-CBOW-FUSED-NS",
    "W2V-PTB-SKIPGRAM-FUSED-NS", "W2V-PTB-CBOW-FULL",
    "W2V-PTB-SKIPGRAM-FULL", "W2V-PTB-CBOW-ONEHOT-FULL",
    "W2V-PTB-SKIPGRAM-ONEHOT-FULL",
)
E02 = StudyDeclaration("e02", tuple(
    condition(
        name.lower(),
        (name,),
        {
            "W2V-PTB-CBOW-NS": ("PTB-CBOW",),
            "W2V-PTB-SKIPGRAM-NS": ("PTB-SKIPGRAM",),
        }.get(name, ()),
    )
    for name in E02_NAMES
))
E03 = StudyDeclaration("e03", (
    condition("small-rnnlm", ("LM-SMALL-RNN",), ("SMALL-RNNLM",), TRAIN_PPL),
))
E04 = StudyDeclaration("e04", (
    condition("lstm-rnnlm", ("LM-LSTM",), ("LSTM-RNNLM",), TEST_PPL),
))
E05 = StudyDeclaration("e05", (
    condition("rnn-recipe", ("LM-RNN-RECIPE",), metric=TEST_PPL),
    condition("lstm-recipe", ("LM-LSTM-RECIPE",), metric=TEST_PPL),
    condition("lstm-tied-recipe", ("LM-LSTM-TIED-RECIPE",), metric=TEST_PPL),
    condition("better-rnnlm", ("LM-BETTER-RECIPE",), ("BETTER-RNNLM",), TEST_PPL),
))
E06 = StudyDeclaration("e06", tuple(
    condition(
        canonical,
        (implemented,),
        () if original is None else (original,),
        EXACT,
    )
    for canonical, implemented, original in (
        ("seq2seq-forward", "SEQA-VAN-FWD", "SEQ2SEQ-FORWARD"),
        ("seq2seq-reverse", "SEQA-VAN-REV", "SEQ2SEQ-REVERSE"),
        ("peeky-forward", "SEQA-PEEKY-FWD", "PEEKY-FORWARD"),
        ("peeky-reverse", "SEQA-PEEKY-REV", "PEEKY-REVERSE"),
        ("attention-forward", "SEQA-ATTN-FWD", None),
        ("attention-reverse", "SEQA-ATTN-REV", None),
    )
))
E07 = StudyDeclaration("e07", tuple(
    condition(canonical, (implemented,), (original,), EXACT)
    for canonical, implemented, original in (
        ("seq2seq-reverse", "SEQD-VAN-REV", "SEQ2SEQ-REVERSE"),
        ("peeky-reverse", "SEQD-PEEKY-REV", "PEEKY-REVERSE"),
        ("attention-reverse", "SEQD-ATTN-REV", "ATTENTION-REVERSE"),
    )
))
E08 = StudyDeclaration("e08", (
    condition("attention-alignment", ("ATTENTION-ALIGNMENT",), ("ATTENTION-ALIGNMENT",), EXACT),
    condition("attention-alignment-greedy", ("ATTENTION-ALIGNMENT-GREEDY",), metric=EXACT),
))
E09 = StudyDeclaration("e09", tuple(
    condition(name.lower(), (name,), metric=EXACT)
    for name in (
        "SEQA-VAN-FWD",
        "SEQA-VAN-REV",
        "SEQA-PEEKY-FWD",
        "SEQA-PEEKY-REV",
        "SEQA-ATTN-FWD",
        "SEQA-ATTN-REV",
        "SEQA-ATTN-PEEKY-FWD",
        "SEQA-ATTN-PEEKY-REV",
    )
))

STUDIES = {item.study_id: item for item in (E01, E02, E03, E04, E05, E06, E07, E08, E09)}
SUMMARY_METRICS = {
    "e01": (SUMMARY_TRAIN_LOSS, SUMMARY_TEST_LOSS),
    "e02": (SUMMARY_BOOK_LOSS,),
    "e03": (SUMMARY_TRAIN_PPL, SUMMARY_TEST_PPL),
    "e04": (SUMMARY_TRAIN_PPL, SUMMARY_TEST_PPL),
    "e05": (SUMMARY_TRAIN_PPL, SUMMARY_VALID_PPL, SUMMARY_TEST_PPL),
    "e06": (SUMMARY_TEST_ACCURACY,),
    "e07": (SUMMARY_TEST_ACCURACY,),
    "e08": (SUMMARY_TRAIN_EXACT, SUMMARY_TEST_EXACT),
    "e09": (SUMMARY_TRAIN_EXACT, SUMMARY_TEST_EXACT),
}
PROTOCOL_EQUIVALENCE = {
    study_id: (("legacy", "book-source-v1"),) for study_id in STUDIES
}
