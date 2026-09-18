from __future__ import annotations

import importlib

import numpy as np
from deepscratch.profiling import SectionRecorder

from dlfs.ds2.original.benchmark import build_word2vec_full_softmax

from .env import _phase


class OriginalWord2Vec:
    def __init__(
        self,
        model_name: str,
        objective_name: str,
        corpus,
        contexts,
        targets,
        backend,
        *,
        one_hot: bool = False,
    ) -> None:
        trainer_module = importlib.import_module("common.trainer")
        optimizer_class = importlib.import_module("common.optimizer").Adam
        self.backend = backend
        self.contexts = backend.xp.asarray(contexts)
        self.targets = backend.xp.asarray(targets)
        vocab_size = int(np.max(corpus)) + 1
        if objective_name == "FullSoftmax":
            kind = "cbow" if model_name == "CBOW" else "skipgram"
            self.model = build_word2vec_full_softmax(
                kind,
                vocab_size,
                100,
                5,
                one_hot=one_hot,
            )
        else:
            module_name = "ch04.cbow" if model_name == "CBOW" else "ch04.skip_gram"
            model_class = getattr(
                importlib.import_module(module_name),
                model_name,
            )
            self.model = model_class(vocab_size, 100, 5, corpus)
        self.optimizer = optimizer_class()
        self.remove_duplicate = trainer_module.remove_duplicate

    def update(self, batch_x, batch_t, recorder: SectionRecorder | None = None):
        with _phase(recorder, "forward"):
            loss = self.model.forward(batch_x, batch_t)
        with _phase(recorder, "backward"):
            self.model.backward()
        with _phase(recorder, "deduplicate_shared_parameters"):
            params, grads = self.remove_duplicate(self.model.params, self.model.grads)
        with _phase(recorder, "optimizer"):
            self.optimizer.update(params, grads)
        return loss
