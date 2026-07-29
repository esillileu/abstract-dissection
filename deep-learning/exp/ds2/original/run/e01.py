"""Original ch03 toy CBOW full-softmax training."""

from pathlib import Path

from exp.original.measurement import OriginalMeasurements

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, importlib, np, save_csv, save_npz, source_imports, to_host


def run(worktree: Path, output: Path, _root: Path) -> None:
    with source_imports(worktree, gpu=True):
        cp = importlib.import_module("cupy")
        util = importlib.import_module("common.util")
        trainer_cls = importlib.import_module("common.trainer").Trainer
        optimizer_cls = importlib.import_module("common.optimizer").Adam
        model_module = importlib.import_module("ch03.simple_cbow")
        # ch03 predates the book's config.GPU switch and imports NumPy directly.
        # Redirect only its array namespace while leaving the source snapshot intact.
        model_module.np = cp
        model_cls = model_module.SimpleCBOW
        np.random.seed(1)
        cp.random.seed(1)
        corpus, word_to_id, id_to_word = util.preprocess(
            "You say goodbye and I say hello."
        )
        contexts, target = util.create_contexts_target(corpus, 1)
        contexts = util.convert_one_hot(contexts, len(word_to_id))
        target = util.convert_one_hot(target, len(word_to_id))
        contexts, target = util.to_gpu(contexts), util.to_gpu(target)
        model = model_cls(len(word_to_id), 5)
        trainer = trainer_cls(model, optimizer_cls())
        measurements = OriginalMeasurements(output)
        with measurements.training():
            trainer.fit(contexts, target, 1000, 3)
        measurements.save(model.params)
        rows = [
            {"plot_index": index, "loss": value, "eval_interval": 20}
            for index, value in enumerate(trainer.loss_list)
        ]
        vectors = to_host(model.word_vecs)
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
        "cupy",
        {"model": "SimpleCBOW", "epochs": 1000, "batch_size": 3, "window": 1},
        COMMON_SOURCES + ("ch03/simple_cbow.py", "ch03/train.py"),
        run,
    ),
)
