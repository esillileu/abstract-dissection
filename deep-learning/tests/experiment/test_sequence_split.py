from __future__ import annotations

from pathlib import Path

import numpy as np

from mlprosection.datasets.sequence import load_sequence


def test_legacy_sequence_split_matches_the_book_randomstate_permutation() -> None:
    data = load_sequence(
        "addition.txt",
        seed=1984,
        split_algorithm="legacy_numpy_randomstate",
    )
    questions = [line.split("_")[0] for line in (Path("src/mlprosection/datasets/addition.txt")).read_text().splitlines()]
    indices = np.arange(len(questions))
    np.random.RandomState(1984).shuffle(indices)
    id_to_char = data["id_to_char"]
    expected = questions[indices[0]]
    actual = "".join(id_to_char[int(value)] for value in data["train"][0][0])

    assert actual == expected


def test_default_sequence_split_remains_available() -> None:
    current = load_sequence("date.txt", seed=3)
    legacy = load_sequence("date.txt", seed=3, split_algorithm="legacy_numpy_randomstate")

    assert current["train"][0].shape == legacy["train"][0].shape
    assert not np.array_equal(current["train"][0], legacy["train"][0])
