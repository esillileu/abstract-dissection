"""Original ch08 date Seq2seq source-selectable conditions."""

from pathlib import Path

from exp.original.measurement import OriginalMeasurements
from exp.original.runtime_context import array_module, budget, master_seed

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, evaluate_seq2seq, importlib, install_ch07_compatibility_aliases, np, save_csv, save_npz, source_imports
from .common import to_device


CONDITIONS = {
    "seq2seq-reverse": ("Seq2seq", "ch07.seq2seq"),
    "peeky-seq2seq-reverse": ("PeekySeq2seq", "ch07.peeky_seq2seq"),
    "attention-seq2seq-reverse": ("AttentionSeq2seq", "ch08.attention_seq2seq"),
}


def _run(name: str, worktree: Path, output: Path, _root: Path) -> None:
    class_name, module_name = CONDITIONS[name]
    with source_imports(worktree, gpu=True):
        cp = array_module()
        sequence = importlib.import_module("dataset.sequence")
        sequence.numpy.int = int
        util = importlib.import_module("common.util")
        optimizer_cls = importlib.import_module("common.optimizer").Adam
        trainer_cls = importlib.import_module("common.trainer").Trainer
        install_ch07_compatibility_aliases()
        model_cls = getattr(importlib.import_module(module_name), class_name)
        (x_train, t_train), (x_test, t_test) = sequence.load_data("date.txt")
        x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
        np.random.seed(master_seed())
        cp.random.seed(master_seed())
        x_train, t_train = to_device(util, x_train), to_device(util, t_train)
        x_test, t_test = to_device(util, x_test), to_device(util, t_test)
        model = model_cls(len(sequence.get_vocab()[0]), 16, 256)
        trainer = trainer_cls(model, optimizer_cls())
        measurements = OriginalMeasurements(output)
        rows = []
        predictions = []
        epochs = budget("max_epochs", 10)
        for epoch in range(epochs):
            with measurements.training():
                trainer.fit(x_train, t_train, 1, 128, max_grad=5.0)
            accuracy, examples = evaluate_seq2seq(
                model, x_test, t_test, reverse=True
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
        f"dlfs2.ch08.date.{name}",
        "cupy",
        {"model": class_name, "reverse": True, "epochs": 10, "batch_size": 128},
        COMMON_SOURCES
        + (
            "ch07/seq2seq.py",
            "ch07/peeky_seq2seq.py",
            "ch08/attention_layer.py",
            "ch08/attention_seq2seq.py",
            "ch08/train.py",
        ),
        lambda worktree, output, root, name=name: _run(
            name, worktree, output, root
        ),
    )
    for name, (class_name, _module_name) in CONDITIONS.items()
)
