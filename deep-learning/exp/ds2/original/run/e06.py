"""Original ch07 addition Seq2seq source-selectable conditions."""

from pathlib import Path

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, evaluate_seq2seq, importlib, np, save_csv, save_npz, source_imports


CONDITIONS = {
    "seq2seq-forward": ("Seq2seq", False),
    "seq2seq-reverse": ("Seq2seq", True),
    "peeky-seq2seq-forward": ("PeekySeq2seq", False),
    "peeky-seq2seq-reverse": ("PeekySeq2seq", True),
}


def _run(name: str, worktree: Path, output: Path, _root: Path) -> None:
    class_name, reverse = CONDITIONS[name]
    with source_imports(worktree):
        sequence = importlib.import_module("dataset.sequence")
        optimizer_cls = importlib.import_module("common.optimizer").Adam
        trainer_cls = importlib.import_module("common.trainer").Trainer
        module = (
            importlib.import_module("ch07.seq2seq")
            if class_name == "Seq2seq"
            else importlib.import_module("ch07.peeky_seq2seq")
        )
        model_cls = getattr(module, class_name)
        (x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
        if reverse:
            x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
        np.random.seed(1)
        model = model_cls(len(sequence.get_vocab()[0]), 16, 128)
        trainer = trainer_cls(model, optimizer_cls())
        rows = []
        predictions = []
        for epoch in range(25):
            trainer.fit(x_train, t_train, 1, 128, max_grad=5.0)
            accuracy, examples = evaluate_seq2seq(
                model, x_test, t_test, reverse=reverse
            )
            rows.append({"epoch": epoch, "accuracy": accuracy})
            if epoch == 24:
                predictions = examples
        params = checkpoint_arrays(model.params)
    save_csv(output / "metrics.csv", rows)
    save_csv(output / "predictions.csv", predictions)
    save_npz(output / "checkpoint.npz", **params)


TRIALS = tuple(
    Trial(
        f"dlfs2.ch07.addition.{name}",
        "numpy",
        {
            "model": class_name,
            "reverse": reverse,
            "epochs": 25,
            "batch_size": 128,
        },
        COMMON_SOURCES
        + ("ch07/seq2seq.py", "ch07/peeky_seq2seq.py", "ch07/train_seq2seq.py"),
        lambda worktree, output, root, name=name: _run(
            name, worktree, output, root
        ),
    )
    for name, (class_name, reverse) in CONDITIONS.items()
)
