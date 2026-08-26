"""Original ch07 addition Seq2seq source-selectable conditions."""

from pathlib import Path

from dlfs.original_runtime.measurement import OriginalMeasurements
from dlfs.original_runtime.runtime_context import array_module, budget, master_seed

from .common import (
    COMMON_SOURCES,
    Trial,
    checkpoint_arrays,
    evaluate_seq2seq,
    importlib,
    install_ch07_compatibility_aliases,
    np,
    save_csv,
    save_npz,
    source_imports,
    to_device,
)

CONDITIONS = {
    "seq2seq-forward": ("Seq2seq", False),
    "seq2seq-reverse": ("Seq2seq", True),
    "peeky-seq2seq-forward": ("PeekySeq2seq", False),
    "peeky-seq2seq-reverse": ("PeekySeq2seq", True),
}


def _run(name: str, worktree: Path, output: Path, _root: Path) -> None:
    class_name, reverse = CONDITIONS[name]
    with source_imports(worktree, gpu=True):
        cp = array_module()
        sequence = importlib.import_module("dataset.sequence")
        sequence.numpy.int = int
        util = importlib.import_module("common.util")
        optimizer_cls = importlib.import_module("common.optimizer").Adam
        trainer_cls = importlib.import_module("common.trainer").Trainer
        install_ch07_compatibility_aliases()
        module = (
            importlib.import_module("ch07.seq2seq")
            if class_name == "Seq2seq"
            else importlib.import_module("ch07.peeky_seq2seq")
        )
        model_cls = getattr(module, class_name)
        (x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
        if reverse:
            x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
        np.random.seed(master_seed())
        cp.random.seed(master_seed())
        x_train, t_train = to_device(util, x_train), to_device(util, t_train)
        x_test, t_test = to_device(util, x_test), to_device(util, t_test)
        model = model_cls(len(sequence.get_vocab()[0]), 16, 128)
        trainer = trainer_cls(model, optimizer_cls())
        measurements = OriginalMeasurements(output)
        rows = []
        predictions = []
        epochs = budget("max_epochs", 25)
        for epoch in range(epochs):
            with measurements.training():
                trainer.fit(x_train, t_train, 1, 128, max_grad=5.0)
            accuracy, examples = evaluate_seq2seq(
                model, x_test, t_test, reverse=reverse
            )
            rows.append({"epoch": epoch, "accuracy": accuracy})
            if epoch == epochs - 1:
                predictions = examples
        measurements.save(model.params)
        params = checkpoint_arrays(model.params)
    save_csv(output / "metrics.csv", rows)
    save_csv(output / "predictions.csv", predictions)
    save_npz(output / "checkpoint.npz", **params)


TRIALS = tuple(
    Trial(
        f"dlfs2.ch07.addition.{name}",
        "cupy",
        {
            "model": class_name,
            "reverse": reverse,
            "epochs": 25,
            "batch_size": 128,
        },
        (
            *COMMON_SOURCES,
            "ch07/seq2seq.py",
            "ch07/peeky_seq2seq.py",
            "ch07/train_seq2seq.py",
        ),
        lambda worktree, output, root, name=name: _run(name, worktree, output, root),
    )
    for name, (class_name, reverse) in CONDITIONS.items()
)
