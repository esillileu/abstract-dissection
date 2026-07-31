"""Original ch04 PTB Word2Vec plus ch03-style full-softmax adaptations."""

from pathlib import Path

from exp.original.measurement import OriginalMeasurements

from .common import COMMON_SOURCES, Trial, checkpoint_arrays, importlib, np, save_csv, save_npz, source_imports, to_host


MODEL_SOURCES = {
    "cbow": ("ch04/cbow.py", "ch03/simple_cbow.py"),
    "skipgram": ("ch04/skip_gram.py", "ch03/simple_skip_gram.py"),
}


def build_full_softmax_model(
    kind: str,
    vocab_size: int,
    hidden_size: int,
    window_size: int,
):
    """Build a ch04 embedding model with the ch03 full-softmax output path."""
    layers = importlib.import_module("common.layers")
    xp = importlib.import_module("common.np").np
    embedding_cls = layers.Embedding
    matmul_cls = layers.MatMul
    loss_cls = layers.SoftmaxWithLoss

    if kind == "cbow":
        class FullSoftmaxCBOW:
            def __init__(self):
                w_in = 0.01 * xp.random.randn(
                    vocab_size, hidden_size
                ).astype("f")
                w_out = 0.01 * xp.random.randn(
                    hidden_size, vocab_size
                ).astype("f")
                self.in_layers = [
                    embedding_cls(w_in) for _ in range(2 * window_size)
                ]
                self.out_layer = matmul_cls(w_out)
                self.loss_layer = loss_cls()
                model_layers = self.in_layers + [self.out_layer]
                self.params = [
                    param
                    for layer in model_layers
                    for param in layer.params
                ]
                self.grads = [
                    grad
                    for layer in model_layers
                    for grad in layer.grads
                ]
                self.word_vecs = w_in

            def forward(self, contexts, target):
                hidden = 0
                for index, layer in enumerate(self.in_layers):
                    hidden += layer.forward(contexts[:, index])
                hidden *= 1 / len(self.in_layers)
                score = self.out_layer.forward(hidden)
                return self.loss_layer.forward(score, target)

            def backward(self, dout=1):
                dh = self.out_layer.backward(self.loss_layer.backward(dout))
                dh *= 1 / len(self.in_layers)
                for layer in self.in_layers:
                    layer.backward(dh)
                return None

        return FullSoftmaxCBOW()

    if kind == "skipgram":
        class FullSoftmaxSkipGram:
            def __init__(self):
                w_in = 0.01 * xp.random.randn(
                    vocab_size, hidden_size
                ).astype("f")
                w_out = 0.01 * xp.random.randn(
                    hidden_size, vocab_size
                ).astype("f")
                self.in_layer = embedding_cls(w_in)
                self.out_layer = matmul_cls(w_out)
                self.loss_layers = [
                    loss_cls() for _ in range(2 * window_size)
                ]
                model_layers = [self.in_layer, self.out_layer]
                self.params = [
                    param
                    for layer in model_layers
                    for param in layer.params
                ]
                self.grads = [
                    grad
                    for layer in model_layers
                    for grad in layer.grads
                ]
                self.word_vecs = w_in

            def forward(self, contexts, target):
                hidden = self.in_layer.forward(target)
                score = self.out_layer.forward(hidden)
                return sum(
                    layer.forward(score, contexts[:, index])
                    for index, layer in enumerate(self.loss_layers)
                )

            def backward(self, dout=1):
                ds = sum(layer.backward(dout) for layer in self.loss_layers)
                self.in_layer.backward(self.out_layer.backward(ds))
                return None

        return FullSoftmaxSkipGram()

    raise ValueError(f"unknown Word2Vec model: {kind}")


def _run(
    kind: str,
    objective: str,
    worktree: Path,
    output: Path,
    _root: Path,
) -> None:
    with source_imports(worktree, gpu=True):
        cp = importlib.import_module("cupy")
        util = importlib.import_module("common.util")
        ptb = importlib.import_module("dataset.ptb")
        trainer_cls = importlib.import_module("common.trainer").Trainer
        optimizer_cls = importlib.import_module("common.optimizer").Adam
        corpus, word_to_id, id_to_word = ptb.load_data("train")
        contexts, target = util.create_contexts_target(corpus, 5)
        contexts, target = util.to_gpu(contexts), util.to_gpu(target)
        cp.random.seed(1)
        np.random.seed(1)
        if objective == "negative-sampling":
            class_name = {"cbow": "CBOW", "skipgram": "SkipGram"}[kind]
            module_name = {
                "cbow": "ch04.cbow",
                "skipgram": "ch04.skip_gram",
            }[kind]
            model_cls = getattr(importlib.import_module(module_name), class_name)
            model = model_cls(len(word_to_id), 100, 5, corpus)
        elif objective == "full-softmax":
            model = build_full_softmax_model(kind, len(word_to_id), 100, 5)
        else:
            raise ValueError(f"unknown Word2Vec objective: {objective}")
        trainer = trainer_cls(model, optimizer_cls())
        measurements = OriginalMeasurements(output)
        with measurements.training():
            trainer.fit(contexts, target, 10, 100)
        measurements.save(model.params)
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
        (
            f"dlfs2.ch04.ptb-{kind}-negative-sampling"
            if objective == "negative-sampling"
            else f"ext.ds2.ptb-{kind}-full-softmax"
        ),
        "cupy",
        (
            {
                "model": {"cbow": "CBOW", "skipgram": "SkipGram"}[kind],
                "epochs": 10,
                "batch_size": 100,
                "window": 5,
            }
            if objective == "negative-sampling"
            else {
                "model": {"cbow": "CBOW", "skipgram": "SkipGram"}[kind],
                "objective": objective,
                "epochs": 10,
                "batch_size": 100,
                "window": 5,
            }
        ),
        (
            COMMON_SOURCES
            + (
                "ch04/cbow.py",
                "ch04/skip_gram.py",
                "ch04/negative_sampling_layer.py",
                "ch04/train.py",
            )
            if objective == "negative-sampling"
            else COMMON_SOURCES
            + MODEL_SOURCES[kind]
            + ("ch04/train.py",)
        ),
        lambda worktree, output, root, kind=kind, objective=objective: _run(
            kind, objective, worktree, output, root
        ),
    )
    for kind in ("cbow", "skipgram")
    for objective in ("negative-sampling", "full-softmax")
)
