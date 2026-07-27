"""Original ch04 PTB Word2Vec conditions through native ``config.GPU``."""

from pathlib import Path

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, importlib, np, save_csv, save_npz, source_imports, to_host


MODELS = {
    "cbow": ("CBOW", "ch04.cbow"),
    "skipgram": ("SkipGram", "ch04.skip_gram"),
}


def _run(kind: str, worktree: Path, output: Path, _root: Path) -> None:
    with source_imports(worktree, gpu=True):
        cp = importlib.import_module("cupy")
        util = importlib.import_module("common.util")
        ptb = importlib.import_module("dataset.ptb")
        trainer_cls = importlib.import_module("common.trainer").Trainer
        optimizer_cls = importlib.import_module("common.optimizer").Adam
        class_name, module_name = MODELS[kind]
        model_cls = getattr(importlib.import_module(module_name), class_name)
        corpus, word_to_id, id_to_word = ptb.load_data("train")
        contexts, target = util.create_contexts_target(corpus, 5)
        contexts, target = util.to_gpu(contexts), util.to_gpu(target)
        cp.random.seed(1)
        np.random.seed(1)
        model = model_cls(len(word_to_id), 100, 5, corpus)
        trainer = trainer_cls(model, optimizer_cls())
        trainer.fit(contexts, target, 10, 100)
        cp.cuda.get_current_stream().synchronize()
        rows = [
            {"plot_index": index, "loss": value, "eval_interval": 20}
            for index, value in enumerate(trainer.loss_list)
        ]
        vectors = to_host(model.word_vecs).astype(np.float16)
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
        f"dlfs2.ch04.ptb-{kind}-negative-sampling",
        "cupy",
        {"model": class_name, "epochs": 10, "batch_size": 100, "window": 5},
        COMMON_SOURCES
        + (
            "ch04/cbow.py",
            "ch04/skip_gram.py",
            "ch04/negative_sampling_layer.py",
            "ch04/train.py",
        ),
        lambda worktree, output, root, kind=kind: _run(
            kind, worktree, output, root
        ),
    )
    for kind, (class_name, _module_name) in MODELS.items()
)
