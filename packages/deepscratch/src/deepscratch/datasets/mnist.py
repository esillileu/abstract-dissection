import gzip
import os
import os.path
import pickle
import urllib.request
from pathlib import Path

import numpy as np

url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        path = Path(data_dir)
    elif env := os.getenv("DEEPSCRATCH_DATA_DIR"):
        path = Path(env) / "mnist" if not Path(env).name == "mnist" else Path(env)
    else:
        path = Path("./data/mnist")
    path.mkdir(parents=True, exist_ok=True)
    return path


save_file = str(resolve_data_dir() / "mnist.pkl")


def _download(file_name, target_dir: Path):
    file_path = target_dir / file_name

    if file_path.exists():
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, str(file_path))
    print("Done")


def download_mnist(data_dir: Path | str | None = None):
    target = resolve_data_dir(data_dir)
    for v in key_file.values():
        _download(v, target)


def _load_label(file_name, target_dir: Path):
    file_path = target_dir / file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name, target_dir: Path):
    file_path = target_dir / file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def _convert_numpy(target_dir: Path):
    dataset = {}
    dataset["train_img"] = _load_img(key_file["train_img"], target_dir)
    dataset["train_label"] = _load_label(key_file["train_label"], target_dir)
    dataset["test_img"] = _load_img(key_file["test_img"], target_dir)
    dataset["test_label"] = _load_label(key_file["test_label"], target_dir)

    return dataset


def init_mnist(data_dir: Path | str | None = None):
    target = resolve_data_dir(data_dir)
    save_file = target / "mnist.pkl"
    download_mnist(target)
    dataset = _convert_numpy(target)
    print("Creating pickle file ...")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(
    normalize=True,
    flatten=True,
    one_hot_label=False,
    gpu=False,
    data_dir: Path | str | None = None,
):
    target = resolve_data_dir(data_dir)
    save_file = target / "mnist.pkl"

    if not save_file.exists():
        init_mnist(target)

    with open(save_file, "rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    from deepscratch.core import Tensor

    if gpu:
        return (
            (
                Tensor(dataset["train_img"], backend="gpu"),
                Tensor(dataset["train_label"], backend="gpu"),
            ),
            (
                Tensor(dataset["test_img"], backend="gpu"),
                Tensor(dataset["test_label"], backend="gpu"),
            ),
        )

    return (
        (
            Tensor(dataset["train_img"], backend="cpu"),
            Tensor(dataset["train_label"], backend="cpu"),
        ),
        (
            Tensor(dataset["test_img"], backend="cpu"),
            Tensor(dataset["test_label"], backend="cpu"),
        ),
    )
