"""Complete DS1 normalized condition and metric declarations."""

from exp.deepscratch.analysis.declarations import (
    ConditionDeclaration,
    MetricDeclaration,
    StudyDeclaration,
)


TRAIN_FULL_ACCURACY = "final/train-full/accuracy"
TRAIN_TEST_ACCURACY_GAP = "final/train-test/accuracy-gap"
ACCURACY = MetricDeclaration(
    "test_accuracy", "fraction", "test", "run",
    ("final/test/accuracy", "test/accuracy"),
    ("final/test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
TRAIN_LOSS = MetricDeclaration(
    "train_loss", "nats", "train", "update",
    ("train/loss", "train/objective"),
    ("train/loss", "train/objective"),
    protocols=("book-source-v1", "legacy"),
)
TRAIN_ACCURACY_CURVE = MetricDeclaration(
    "train_accuracy_curve", "fraction", "train", "epoch",
    ("update/eval_train/accuracy", "train/accuracy"),
    ("update/eval_train/accuracy", "train/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
TEST_ACCURACY_CURVE = MetricDeclaration(
    "test_accuracy_curve", "fraction", "test", "epoch",
    ("update/eval_test/accuracy", "test/accuracy"),
    ("update/eval_test/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TRAIN_ACCURACY = MetricDeclaration(
    "train_accuracy", "fraction", "train", "run",
    ("final/train-full/accuracy", "final/train/accuracy", "train/accuracy"),
    ("final/train-full/accuracy", "final/train/accuracy", "train/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TEST_ACCURACY = MetricDeclaration(
    "test_accuracy", "fraction", "test", "run",
    ("final/test/accuracy", "final/test-full/accuracy", "test/accuracy"),
    ("final/test/accuracy", "final/test-full/accuracy", "test/accuracy"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TRAIN_LOSS = MetricDeclaration(
    "train_loss", "nats", "train", "run",
    ("final/train/loss", "train/loss", "train/objective"),
    ("final/train/loss", "train/loss", "train/objective"),
    protocols=("book-source-v1", "legacy"),
)
SUMMARY_TEST_LOSS = MetricDeclaration(
    "test_loss", "nats", "test", "run",
    ("final/test/loss", "test/loss"),
    ("final/test/loss", "test/loss"),
    protocols=("book-source-v1", "legacy"),
)


def condition(canonical, implemented, original=(), metrics=(ACCURACY,)):
    return ConditionDeclaration(
        canonical, tuple(implemented), tuple(original), tuple(metrics)
    )


E01 = StudyDeclaration("e01", tuple(
    condition(
        f"optimizer.{name.lower()}",
        (f"MLP-OPT-{name}",),
        (f"OPT-{name}",),
        (ACCURACY, TRAIN_LOSS),
    )
    for name in ("SGD", "MOMENTUM", "ADAGRAD", "ADAM")
))
E02 = StudyDeclaration("e02", tuple(
    condition(
        f"initializer.{canonical}",
        (f"MLP-INIT-{implemented}",),
        (f"INIT-{implemented}",),
        (ACCURACY, TRAIN_LOSS),
    )
    for canonical, implemented in (("std001", "STD001"), ("xavier", "XAVIER"), ("he", "HE"))
))
E03 = StudyDeclaration("e03", (
    condition("weight-decay.off", ("REG-WD-OFF",), metrics=(ACCURACY, TRAIN_ACCURACY_CURVE, TEST_ACCURACY_CURVE)),
    condition("weight-decay.0.1", ("REG-WD-01",), ("WEIGHT-DECAY-01",), (ACCURACY, TRAIN_ACCURACY_CURVE, TEST_ACCURACY_CURVE)),
))
E04 = StudyDeclaration("e04", (
    condition("dropout.off", ("REG-DROPOUT-OFF",), metrics=(ACCURACY, TRAIN_ACCURACY_CURVE, TEST_ACCURACY_CURVE)),
    condition("dropout.0.2", ("REG-DROPOUT-ON-02",), ("DROPOUT-02",), (ACCURACY, TRAIN_ACCURACY_CURVE, TEST_ACCURACY_CURVE)),
))
E05 = StudyDeclaration("e05", tuple(
    condition(
        f"batchnorm.scale-{scale:02d}.{state.lower()}",
        (f"BN-SCALE-{scale:02d}-{state}",),
        (f"BN-{scale:02d}-{state}",),
        (ACCURACY, TRAIN_ACCURACY_CURVE),
    )
    for scale in range(1, 17) for state in ("OFF", "ON")
))
E06 = StudyDeclaration("e06", (
    condition("simple-cnn", ("CNN-SIMPLE-BOOK",), ("SIMPLE-CONVNET",), (ACCURACY, TRAIN_ACCURACY_CURVE, TEST_ACCURACY_CURVE)),
))
E07 = StudyDeclaration("e07", (
    condition("deep-cnn", ("CNN-DEEP-BOOK",), ("DEEP-CONVNET",), (ACCURACY, TRAIN_ACCURACY_CURVE, TEST_ACCURACY_CURVE)),
))
E08 = StudyDeclaration("e08", tuple(
    condition(name.lower().replace("_", "-"), (name,))
    for name in (
        "NN-MATCHED", "NN-MATCHED-PERMUTED", "CNN-SIMPLE-SPATIAL",
        "CNN-SIMPLE-SPATIAL-PERMUTED",
    )
))
E09 = StudyDeclaration("e09", tuple(
    condition(
        f"trajectory.{name.lower()}",
        (f"TOY-{name}",),
        (f"PATH-{name}",),
        (),
    )
    for name in ("SGD", "MOMENTUM", "ADAGRAD", "ADAM")
))
E10 = StudyDeclaration("e10", tuple(
    condition(
        f"activation.{activation.lower()}.{initializer.lower()}",
        (f"ACT-{activation}-{initializer}",),
        ("ACTIVATION-SIGMOID",) if activation == "SIGMOID" and initializer == "STD001" else (),
        (),
    )
    for activation in ("SIGMOID", "TANH", "RELU")
    for initializer in ("STD001", "XAVIER", "HE", "STD1")
))
E12 = StudyDeclaration("e12", (
    condition("extended-mlp", ("MLP-EXT-ALL-BOOK",)),
))

STUDIES = {item.study_id: item for item in (E01, E02, E03, E04, E05, E06, E07, E08, E09, E10, E12)}
SUMMARY_METRICS = {
    study_id: (SUMMARY_TRAIN_ACCURACY, SUMMARY_TEST_ACCURACY)
    for study_id in ("e03", "e04", "e05", "e06", "e07", "e08", "e12")
}
SUMMARY_METRICS.update({
    "e01": (SUMMARY_TRAIN_LOSS, SUMMARY_TEST_LOSS),
    "e02": (SUMMARY_TRAIN_LOSS, SUMMARY_TEST_LOSS),
    "e09": (SUMMARY_TRAIN_LOSS, SUMMARY_TEST_LOSS),
    "e10": (),
})
PROTOCOL_EQUIVALENCE = {
    study_id: (("legacy", "book-source-v1"),) for study_id in STUDIES
}
