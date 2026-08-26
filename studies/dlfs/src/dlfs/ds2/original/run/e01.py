"""Original ch03 toy CBOW and Skip-gram full-softmax training."""

from pathlib import Path

from dlfs.original_runtime.measurement import OriginalMeasurements
from dlfs.original_runtime.runtime_context import array_module, budget, master_seed

from .common import (
    COMMON_SOURCES,
    Trial,
    checkpoint_arrays,
    importlib,
    np,
    save_csv,
    save_npz,
    source_imports,
    to_device,
    to_host,
)

MODEL_SOURCES = {
    "cbow": "simple_cbow",
    "skipgram": "simple_skip_gram",
}


def run(kind: str, worktree: Path, output: Path, _root: Path) -> None:
    module_name = MODEL_SOURCES[kind]
    class_name = {"cbow": "SimpleCBOW", "skipgram": "SimpleSkipGram"}[kind]
    with source_imports(worktree, gpu=True):
        cp = array_module()
        util = importlib.import_module("common.util")
        trainer_cls = importlib.import_module("common.trainer").Trainer
        optimizer_cls = importlib.import_module("common.optimizer").Adam
        model_module = importlib.import_module(f"ch03.{module_name}")
        # ch03 predates the book's config.GPU switch and imports NumPy directly.
        # Redirect only its array namespace while leaving the source snapshot intact.
        model_module.np = cp
        model_cls = getattr(model_module, class_name)
        np.random.seed(master_seed())
        cp.random.seed(master_seed())
        corpus, word_to_id, id_to_word = util.preprocess(
            "You say goodbye and I say hello."
        )
        contexts, target = util.create_contexts_target(corpus, 1)
        contexts = util.convert_one_hot(contexts, len(word_to_id))
        target = util.convert_one_hot(target, len(word_to_id))
        contexts, target = to_device(util, contexts), to_device(util, target)
        model = model_cls(len(word_to_id), 5)
        trainer = trainer_cls(model, optimizer_cls())
        measurements = OriginalMeasurements(output)
        with measurements.training():
            trainer.fit(contexts, target, budget("max_epochs", 1000), 3)
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


TRIALS = tuple(
    Trial(
        (
            "dlfs2.ch03.toy-cbow-full-softmax"
            if kind == "cbow"
            else "dlfs2.ch03.toy-skipgram-full-softmax"
        ),
        "cupy",
        {
            "model": {"cbow": "SimpleCBOW", "skipgram": "SimpleSkipGram"}[kind],
            "epochs": 1000,
            "batch_size": 3,
            "window": 1,
        },
        (*COMMON_SOURCES, f"ch03/{MODEL_SOURCES[kind]}.py", "ch03/train.py"),
        lambda worktree, output, root, kind=kind: run(kind, worktree, output, root),
    )
    for kind in ("cbow", "skipgram")
)
