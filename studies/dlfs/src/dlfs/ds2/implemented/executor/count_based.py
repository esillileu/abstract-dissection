from __future__ import annotations

import json
from dataclasses import asdict
from time import perf_counter

import numpy as np
from deepscratch.core import configure_runtime

from dlfs.ds2.implemented.adapters import load_ds2_word2vec_corpus
from dlfs.ds2.statistical import (
    create_cooccurrence_matrix,
    factorize_ppmi,
    positive_pmi,
)
from repro_core.context import ExperimentContext
from repro_core.context.contracts import ExperimentResult

from .checkpoint import _publish_array_checkpoint
from .common import _artifact_root, _final, _mapping


class CountBasedEmbeddingExecutor:
    """Build and persist PTB PPMI representations and their factorizations."""

    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        _backend, streams, runtime = configure_runtime(config)
        context.metadata.update({"runtime": runtime, "seed_streams": asdict(streams)})
        dataset = _mapping(config, "dataset")
        model = _mapping(config, "model")
        corpus, word_to_id = load_ds2_word2vec_corpus(dataset)
        window_size = int(dataset.get("window_size", 2))
        components = min(int(model.get("embedding_size", 100)), len(word_to_id))
        method = str(model.get("name", "ppmi")).lower()
        seed = int(config.get("seed", 1))

        started = perf_counter()
        cooccurrence = create_cooccurrence_matrix(corpus, len(word_to_id), window_size)
        cooccurrence_s = perf_counter() - started
        started = perf_counter()
        ppmi = positive_pmi(cooccurrence)
        ppmi_s = perf_counter() - started
        started = perf_counter()
        vectors, singular_values, right_factors = factorize_ppmi(
            ppmi,
            method=method,
            components=components,
            seed=seed,
            n_iter=int(model.get("n_iter", 5)),
        )
        decomposition_s = perf_counter() - started
        total_s = cooccurrence_s + ppmi_s + decomposition_s

        artifact_root = _artifact_root(config, context)
        artifact_root.mkdir(parents=True, exist_ok=True)
        matrix_path = artifact_root / "statistical_matrices.npz"
        payload = {
            "cooccurrence": cooccurrence,
            "ppmi": ppmi,
            "word_vectors": vectors,
            "singular_values": singular_values,
        }
        if right_factors is not None:
            payload["right_factors"] = right_factors
        np.savez_compressed(matrix_path, **payload)
        timing_path = artifact_root / "timing.json"
        timing = {
            "cooccurrence_s": cooccurrence_s,
            "ppmi_s": ppmi_s,
            "decomposition_s": decomposition_s,
            "total_s": total_s,
            "method": method,
        }
        timing_path.write_text(
            json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checkpoint = _publish_array_checkpoint(
            context,
            W_in=vectors,
            word_vectors=vectors,
            singular_values=singular_values,
        )
        metrics = _final(
            updates=0,
            epochs=0,
            samples=len(corpus),
            **{
                "final/runtime/cooccurrence_s": cooccurrence_s,
                "final/runtime/ppmi_s": ppmi_s,
                "final/runtime/decomposition_s": decomposition_s,
                "runtime/train_total_s": total_s,
            },
        )
        return ExperimentResult(
            metrics=metrics,
            artifact_root=artifact_root,
            artifacts=(matrix_path, timing_path, checkpoint),
            metric_rows=tuple(
                (0, name, value)
                for name, value in (
                    ("runtime/cooccurrence_s", cooccurrence_s),
                    ("runtime/ppmi_s", ppmi_s),
                    ("runtime/decomposition_s", decomposition_s),
                    ("runtime/total_s", total_s),
                )
            ),
        )
