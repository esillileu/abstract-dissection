"""Persist the five original attention maps from the e07 host checkpoint."""

from pathlib import Path

from exp.original.cache import cache_is_valid

from .common import COMMON_SOURCES, Trial, importlib, load_npz, np, restore_params, save_csv, save_npz, source_imports


ATTENTION_TRIAL = "dlfs2.ch08.date.attention-seq2seq-reverse"


def run(worktree: Path, output: Path, root: Path) -> None:
    checkpoint_dir = root / "data" / "e07" / ATTENTION_TRIAL
    if not cache_is_valid(checkpoint_dir):
        raise ValueError(
            "e08 requires cached e07 attention checkpoint: "
            f"e07/{ATTENTION_TRIAL}"
        )
    archive = load_npz(checkpoint_dir / "checkpoint.npz")
    with source_imports(worktree):
        sequence = importlib.import_module("dataset.sequence")
        model_cls = importlib.import_module(
            "ch08.attention_seq2seq"
        ).AttentionSeq2seq
        (_, _), (x_test, t_test) = sequence.load_data("date.txt")
        x_test = x_test[:, ::-1]
        char_to_id, id_to_char = sequence.get_vocab()
        model = model_cls(len(char_to_id), 16, 256)
        restore_params(model.params, archive)
        np.random.seed(1984)
        arrays = {}
        labels = []
        for example in range(5):
            index = int(np.random.randint(0, len(x_test)))
            x = x_test[[index]]
            t = t_test[[index]]
            model.forward(x, t)
            weights = np.asarray(model.decoder.attention.attention_weights)
            attention = weights.reshape(weights.shape[0], weights.shape[2])[:, ::-1]
            source = x[:, ::-1]
            arrays[f"attention_{example}"] = attention
            labels.append(
                {
                    "example": example,
                    "dataset_index": index,
                    "row_labels": "".join(id_to_char[int(i)] for i in source[0]),
                    "column_labels": "".join(
                        id_to_char[int(i)] for i in t[0][1:]
                    ),
                }
            )
    save_npz(output / "attention.npz", **arrays)
    save_csv(output / "labels.csv", labels)


TRIALS = (
    Trial(
        "dlfs2.ch08.attention-alignment",
        "numpy",
        {"selection_seed": 1984, "examples": 5, "checkpoint_trial": ATTENTION_TRIAL},
        COMMON_SOURCES
        + (
            "ch08/attention_layer.py",
            "ch08/attention_seq2seq.py",
            "ch08/visualize_attention.py",
        ),
        run,
    ),
)
