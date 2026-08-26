import os
from pathlib import Path

import numpy as np


class SequenceDataset(dict):
    """Dictionary-like dataset split container that also supports tuple unpacking."""

    def __iter__(self):
        # Allow unpacking as: (x_train, t_train), (x_test, t_test), (char_to_id, id_to_char)
        return iter(
            (self["train"], self["test"], (self["char_to_id"], self["id_to_char"]))
        )


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        path = Path(data_dir)
    elif env := os.getenv("DEEPSCRATCH_DATA_DIR"):
        path = Path(env) / "sequence" if not Path(env).name == "sequence" else Path(env)
    else:
        path = Path("./data/sequence")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_data(
    file_name="addition.txt",
    seed=1984,
    data_dir: Path | str | None = None,
    split_algorithm="default_rng",
    **kwargs,
):
    target = resolve_data_dir(data_dir)
    file_path = target / file_name

    if not file_path.exists():
        # Fallback search
        root_data = Path("./data/sequence") / file_name
        if root_data.exists():
            file_path = root_data
        else:
            raise FileNotFoundError(f"Sequence dataset not found at {file_path}")

    questions = []
    answers = []

    for line in open(file_path, encoding="utf-8"):
        idx = line.find("_")
        questions.append(line[:idx])
        answers.append(line[idx:-1])

    char_to_id = {}
    id_to_char = {}

    def _update_vocab(txt):
        chars = list(txt)
        for val in chars:
            if val not in char_to_id:
                idx = len(char_to_id)
                char_to_id[val] = idx
                id_to_char[idx] = val

    for i in range(len(questions)):
        q = questions[i]
        a = answers[i]
        _update_vocab(q)
        _update_vocab(a)

    x = np.zeros((len(questions), len(questions[0])), dtype=int)
    t = np.zeros((len(answers), len(answers[0])), dtype=int)

    for i, sentence in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        t[i] = [char_to_id[c] for c in list(sentence)]

    indices = np.arange(len(x))
    if seed is not None:
        if split_algorithm == "legacy_randint":
            np.random.seed(seed)
        else:
            np.random.seed(seed)
    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return SequenceDataset(
        {
            "train": (x_train, t_train),
            "test": (x_test, t_test),
            "char_to_id": char_to_id,
            "id_to_word": id_to_char,
            "id_to_char": id_to_char,
        }
    )


load_sequence = load_data
