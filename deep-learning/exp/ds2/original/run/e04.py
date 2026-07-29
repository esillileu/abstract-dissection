"""Original ch06 PTB LSTM Rnnlm through the book's CuPy source path."""

from pathlib import Path

from exp.original.measurement import OriginalMeasurements

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, importlib, np, save_csv, save_npz, source_imports


def run(worktree: Path, output: Path, _root: Path) -> None:
    with source_imports(worktree, gpu=True):
        cp = importlib.import_module("cupy")
        ptb = importlib.import_module("dataset.ptb")
        util = importlib.import_module("common.util")
        model_cls = importlib.import_module("ch06.rnnlm").Rnnlm
        trainer_cls = importlib.import_module("common.trainer").RnnlmTrainer
        optimizer_cls = importlib.import_module("common.optimizer").SGD
        corpus, word_to_id, _ = ptb.load_data("train")
        corpus_test, _, _ = ptb.load_data("test")
        np.random.seed(1)
        cp.random.seed(1)
        corpus = util.to_gpu(corpus)
        corpus_test = util.to_gpu(corpus_test)
        model = model_cls(len(word_to_id), 100, 100)
        trainer = trainer_cls(model, optimizer_cls(20.0))
        measurements = OriginalMeasurements(output)
        with measurements.training():
            trainer.fit(
                corpus[:-1],
                corpus[1:],
                4,
                20,
                35,
                0.25,
                eval_interval=20,
            )
        measurements.save(model.params)
        model.reset_state()
        test_ppl = float(util.eval_perplexity(model, corpus_test))
        rows = [
            {
                "plot_index": index,
                "split": "train",
                "perplexity": value,
                "eval_interval": 20,
            }
            for index, value in enumerate(trainer.ppl_list)
        ]
        rows.append(
            {
                "plot_index": len(rows),
                "split": "test",
                "perplexity": test_ppl,
                "eval_interval": "",
            }
        )
        params = checkpoint_arrays(model.params)
    save_csv(output / "metrics.csv", rows)
    save_npz(output / "checkpoint.npz", **params)


TRIALS = (
    Trial(
        "dlfs2.ch06.ptb-lstm-rnnlm",
        "cupy",
        {
            "epochs": 4,
            "batch_size": 20,
            "time_size": 35,
            "learning_rate": 20.0,
            "max_grad": 0.25,
        },
        COMMON_SOURCES + ("ch06/rnnlm.py", "ch06/train_rnnlm.py"),
        run,
    ),
)
