"""Original ch08 date Seq2seq source-selectable conditions."""

from pathlib import Path

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, evaluate_seq2seq, importlib, np, save_csv, save_npz, source_imports


CONDITIONS = {
    "seq2seq-reverse": ("Seq2seq", "ch07.seq2seq"),
    "peeky-seq2seq-reverse": ("PeekySeq2seq", "ch07.peeky_seq2seq"),
    "attention-seq2seq-reverse": ("AttentionSeq2seq", "ch08.attention_seq2seq"),
}


def _run(name: str, worktree: Path, output: Path, _root: Path) -> None:
    class_name, module_name = CONDITIONS[name]
    with source_imports(worktree):
        sequence = importlib.import_module("dataset.sequence")
        optimizer_cls = importlib.import_module("common.optimizer").Adam
        trainer_cls = importlib.import_module("common.trainer").Trainer
        model_cls = getattr(importlib.import_module(module_name), class_name)
        (x_train, t_train), (x_test, t_test) = sequence.load_data("date.txt")
        x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
        np.random.seed(1)
        model = model_cls(len(sequence.get_vocab()[0]), 16, 256)
        trainer = trainer_cls(model, optimizer_cls())
        rows = []
        predictions = []
        for epoch in range(10):
            trainer.fit(x_train, t_train, 1, 128, max_grad=5.0)
            accuracy, examples = evaluate_seq2seq(
                model, x_test, t_test, reverse=True
            )
            rows.append({"epoch": epoch, "accuracy": accuracy})
            if epoch == 9:
                predictions = examples
        params = checkpoint_arrays(model.params)
    save_csv(output / "metrics.csv", rows)
    save_csv(output / "predictions.csv", predictions)
    save_npz(output / "checkpoint.npz", **params)


TRIALS = tuple(
    Trial(
        f"dlfs2.ch08.date.{name}",
        "numpy",
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
