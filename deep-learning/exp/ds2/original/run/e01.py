"""Original ch03 toy CBOW full-softmax training."""

from pathlib import Path

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, importlib, np, save_csv, save_npz, source_imports


def run(worktree: Path, output: Path, _root: Path) -> None:
    with source_imports(worktree):
        util = importlib.import_module("common.util")
        trainer_cls = importlib.import_module("common.trainer").Trainer
        optimizer_cls = importlib.import_module("common.optimizer").Adam
        model_cls = importlib.import_module("ch03.simple_cbow").SimpleCBOW
        np.random.seed(1)
        corpus, word_to_id, id_to_word = util.preprocess(
            "You say goodbye and I say hello."
        )
        contexts, target = util.create_contexts_target(corpus, 1)
        contexts = util.convert_one_hot(contexts, len(word_to_id))
        target = util.convert_one_hot(target, len(word_to_id))
        model = model_cls(len(word_to_id), 5)
        trainer = trainer_cls(model, optimizer_cls())
        trainer.fit(contexts, target, 1000, 3)
        rows = [
            {"plot_index": index, "loss": value, "eval_interval": 20}
            for index, value in enumerate(trainer.loss_list)
        ]
        vectors = np.asarray(model.word_vecs)
        params = checkpoint_arrays(model.params)
    save_csv(output / "metrics.csv", rows)
    save_csv(
        output / "vocabulary.csv",
        (
            {"word_id": word_id, "word": id_to_word[word_id]}
            for word_id in sorted(id_to_word)
        ),
    )
    save_npz(output / "checkpoint.npz", word_vectors=vectors, **params)


TRIALS = (
    Trial(
        "dlfs2.ch03.toy-cbow-full-softmax",
        "numpy",
        {"model": "SimpleCBOW", "epochs": 1000, "batch_size": 3, "window": 1},
        COMMON_SOURCES + ("ch03/simple_cbow.py", "ch03/train.py"),
        run,
    ),
)
