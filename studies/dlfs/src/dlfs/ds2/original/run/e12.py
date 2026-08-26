"""Original chapter-2 PTB PPMI and SVD experiments."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from sklearn.utils.extmath import randomized_svd

from dlfs.original_runtime.runtime_context import master_seed

from .common import Trial, importlib, np, save_csv, save_npz, source_imports

SOURCE_FILES = (
    "common/util.py",
    "dataset/ptb.py",
    "ch02/count_method_big.py",
)


def _run(method: str, worktree: Path, output: Path, _root: Path) -> None:
    with source_imports(worktree):
        util = importlib.import_module("common.util")
        ptb = importlib.import_module("dataset.ptb")
        corpus, word_to_id, _id_to_word = ptb.load_data("train")

        started = perf_counter()
        cooccurrence = util.create_co_matrix(corpus, len(word_to_id), 2)
        cooccurrence_s = perf_counter() - started
        started = perf_counter()
        ppmi = util.ppmi(cooccurrence, verbose=False)
        ppmi_s = perf_counter() - started
        started = perf_counter()
        singular_values = np.empty(0, dtype=np.float32)
        right_factors = None
        if method == "ppmi":
            vectors = ppmi
        elif method == "svd":
            u, singular_values, _ = np.linalg.svd(ppmi, full_matrices=False)
            vectors = u[:, :100]
        elif method == "randomized_svd":
            vectors, singular_values, right_factors = randomized_svd(
                ppmi,
                n_components=100,
                n_iter=5,
                random_state=master_seed(),
            )
        else:
            raise ValueError(f"unknown count-based factorization: {method}")
        decomposition_s = perf_counter() - started

    total_s = cooccurrence_s + ppmi_s + decomposition_s
    matrices = {
        "cooccurrence": cooccurrence,
        "ppmi": ppmi,
        "word_vectors": vectors.astype(np.float32, copy=False),
        "singular_values": singular_values.astype(np.float32, copy=False),
    }
    if right_factors is not None:
        matrices["right_factors"] = right_factors.astype(np.float32, copy=False)
    save_npz(output / "statistical_matrices.npz", **matrices)
    save_npz(
        output / "checkpoint.npz",
        word_vectors=matrices["word_vectors"],
        singular_values=matrices["singular_values"],
    )
    save_csv(
        output / "metrics.csv",
        (
            {"metric": key, "value": value}
            for key, value in (
                ("cooccurrence_s", cooccurrence_s),
                ("ppmi_s", ppmi_s),
                ("decomposition_s", decomposition_s),
                ("total_s", total_s),
            )
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "timing.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "count_based_pipeline",
                "training_wall_time_s": total_s,
                "cooccurrence_s": cooccurrence_s,
                "ppmi_s": ppmi_s,
                "decomposition_s": decomposition_s,
                "total_s": total_s,
                "method": method,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


TRIALS = tuple(
    Trial(
        f"dlfs2.ch02.ptb-{method.replace('_', '-')}",
        "numpy",
        {
            "method": method,
            "window": 2,
            "embedding_size": None if method == "ppmi" else 100,
            "random_state": "master_seed" if method == "randomized_svd" else None,
        },
        SOURCE_FILES,
        lambda worktree, output, root, method=method: _run(
            method, worktree, output, root
        ),
    )
    for method in ("ppmi", "svd", "randomized_svd")
)
