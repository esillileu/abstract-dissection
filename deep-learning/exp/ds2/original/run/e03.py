"""Original ch05 small-corpus SimpleRnnlm."""

from pathlib import Path

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, importlib, np, save_csv, save_npz, source_imports


def run(worktree: Path, output: Path, _root: Path) -> None:
    with source_imports(worktree):
        ptb = importlib.import_module("dataset.ptb")
        model_cls = importlib.import_module("ch05.simple_rnnlm").SimpleRnnlm
        trainer_cls = importlib.import_module("common.trainer").RnnlmTrainer
        optimizer_cls = importlib.import_module("common.optimizer").SGD
        corpus, _, _ = ptb.load_data("train")
        corpus = corpus[:1000]
        np.random.seed(1)
        model = model_cls(int(max(corpus) + 1), 100, 100)
        trainer = trainer_cls(model, optimizer_cls(0.1))
        trainer.fit(corpus[:-1], corpus[1:], 100, 10, 5)
        rows = [
            {"plot_index": index, "perplexity": value, "eval_interval": 20}
            for index, value in enumerate(trainer.ppl_list)
        ]
        params = checkpoint_arrays(model.params)
    save_csv(output / "metrics.csv", rows)
    save_npz(output / "checkpoint.npz", **params)


TRIALS = (
    Trial(
        "dlfs2.ch05.ptb-small-rnnlm",
        "numpy",
        {"epochs": 100, "batch_size": 10, "time_size": 5, "learning_rate": 0.1},
        COMMON_SOURCES + ("ch05/simple_rnnlm.py", "ch05/train.py"),
        run,
    ),
)
