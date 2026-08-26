import os
import pickle
import urllib.request
from pathlib import Path

import numpy as np

url_base = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
key_file = {"train": "ptb.train.txt", "test": "ptb.test.txt", "valid": "ptb.valid.txt"}
save_file = {"train": "ptb.train.npy", "test": "ptb.test.npy", "valid": "ptb.valid.npy"}
vocab_file = "ptb.vocab.pkl"


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        path = Path(data_dir)
    elif env := os.getenv("DEEPSCRATCH_DATA_DIR"):
        path = Path(env) / "ptb" if not Path(env).name == "ptb" else Path(env)
    else:
        path = Path("./data/ptb")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(file_name, target_dir: Path):
    file_path = target_dir / file_name
    if file_path.exists():
        return

    print("Downloading " + file_name + " ... ")
    try:
        urllib.request.urlretrieve(url_base + file_name, str(file_path))
    except urllib.error.URLError:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url_base + file_name, str(file_path))
    print("Done")


def load_vocab(data_dir: Path | str | None = None):
    target = resolve_data_dir(data_dir)
    vocab_path = target / vocab_file

    if vocab_path.exists():
        with open(vocab_path, "rb") as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    data_type = "train"
    file_name = key_file[data_type]
    file_path = target / file_name

    _download(file_name, target)

    words = (
        open(file_path, encoding="utf-8").read().replace("\n", "<eos>").strip().split()
    )

    for _i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, "wb") as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_data(data_type="train", data_dir: Path | str | None = None):
    """Load PTB corpus split."""
    if data_type == "val":
        data_type = "valid"

    target = resolve_data_dir(data_dir)
    save_path = target / save_file[data_type]
    word_to_id, id_to_word = load_vocab(target)

    if save_path.exists():
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word

    file_name = key_file[data_type]
    file_path = target / file_name
    _download(file_name, target)

    words = (
        open(file_path, encoding="utf-8").read().replace("\n", "<eos>").strip().split()
    )
    corpus = np.array([word_to_id[w] for w in words])

    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word


def load_ptb(*, allow_download: bool = True, data_dir: Path | str | None = None):
    """Load the repository-cached PTB splits as a mapping."""
    target = resolve_data_dir(data_dir)
    train, word_to_id, id_to_word = load_data("train", target)
    valid, _, _ = load_data("valid", target)
    test, _, _ = load_data("test", target)
    return {
        "train": train,
        "valid": valid,
        "test": test,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
    }
