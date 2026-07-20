"""Word2vec models with full-softmax and negative-sampling objectives."""

from __future__ import annotations

from typing import Any

from mlprosection import Tensor
from mlprosection.core.backend import Backend, resolve_backend
from mlprosection.nn.layers import Layer
from mlprosection.nn.sampling import UnigramSampler
from mlprosection.nn.types import Parameter


class Word2Vec(Layer):
    """CBOW or Skip-gram embedding model with a selectable output objective."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        *,
        architecture: str = "cbow",
        objective: str = "negative_sampling",
        negative_samples: int = 5,
        sampler: UnigramSampler | None = None,
        backend: Backend | str | None = None,
    ) -> None:
        resolved = resolve_backend(backend)
        super().__init__(resolved)
        if architecture not in {"cbow", "skipgram"}:
            raise ValueError("architecture must be 'cbow' or 'skipgram'")
        if objective not in {"full_softmax", "negative_sampling"}:
            raise ValueError("objective must be 'full_softmax' or 'negative_sampling'")
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.architecture = architecture
        self.objective = objective
        self.negative_samples = negative_samples
        xp = resolved.xp
        self.W_in = Parameter((0.01 * xp.random.randn(vocab_size, embedding_size)).astype(resolved.float_dtype), backend=resolved, name="W_in")
        self.W_out = Parameter((0.01 * xp.random.randn(vocab_size, embedding_size)).astype(resolved.float_dtype), backend=resolved, name="W_out")
        self.sampler: UnigramSampler | None = None
        if objective == "negative_sampling":
            self.sampler = sampler or UnigramSampler.uniform(vocab_size, backend=resolved)
            if self.sampler.vocab_size != vocab_size:
                raise ValueError("sampler vocabulary does not match the model vocabulary")
            if self.sampler.backend.device != resolved.device:
                raise ValueError("sampler backend must match the model backend")
        self.cache: list[tuple[Any, Any, Any, Any]] = []

    @property
    def word_vectors(self):
        return self.W_in.data

    def forward_manual(self, xs: Tensor, ts: Tensor) -> Tensor:
        xp = self.backend.xp
        indices = xs.data.astype(xp.int64, copy=False)
        targets = ts.data.astype(xp.int64, copy=False)
        self.cache = []
        if self.architecture == "cbow":
            hidden = self.W_in.data[indices].mean(axis=1)
            loss = self._objective(hidden, targets.reshape(-1), indices)
        else:
            centers = indices.reshape(-1)
            context_targets = targets if targets.ndim == 2 else targets[:, None]
            losses = [self._objective(self.W_in.data[centers], context_targets[:, column], centers) for column in range(context_targets.shape[1])]
            loss = sum(losses) / len(losses)
        return Tensor(xp.asarray(loss, dtype=self.backend.float_dtype), backend=self.backend)

    def _objective(self, hidden, labels, source) -> float:
        xp = self.backend.xp
        if self.objective == "full_softmax":
            scores = hidden @ self.W_out.data.T
            scores -= scores.max(axis=1, keepdims=True)
            probabilities = xp.exp(scores)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            loss = -xp.log(probabilities[xp.arange(len(labels)), labels] + 1e-7).mean()
            self.cache.append((hidden, labels, source, probabilities))
            return float(loss)
        if self.sampler is None:
            raise RuntimeError("negative-sampling objective requires a sampler")
        negatives = self.sampler.sample(labels, sample_size=self.negative_samples)
        candidates = xp.concatenate((labels[:, None], negatives), axis=1)
        scores = xp.sum(hidden[:, None, :] * self.W_out.data[candidates], axis=2)
        targets = xp.zeros_like(scores)
        targets[:, 0] = 1
        probabilities = 1 / (1 + xp.exp(-scores))
        loss = -(targets * xp.log(probabilities + 1e-7) + (1 - targets) * xp.log(1 - probabilities + 1e-7)).mean()
        self.cache.append((hidden, candidates, source, (probabilities, targets)))
        return float(loss)

    def backward_manual(self, dout: Tensor | None = None) -> None:
        xp = self.backend.xp
        self.W_in.grad[...] = 0
        self.W_out.grad[...] = 0
        scale = 1.0 if dout is None else float(dout.data)
        for hidden, labels_or_candidates, source, values in self.cache:
            if self.objective == "full_softmax":
                labels, probabilities = labels_or_candidates, values
                gradient = probabilities.copy()
                gradient[xp.arange(len(labels)), labels] -= 1
                gradient *= scale / len(labels)
                self.W_out.grad += gradient.T @ hidden
                dhidden = gradient @ self.W_out.data
            else:
                candidates = labels_or_candidates
                probabilities, targets = values
                gradient = (probabilities - targets) * scale / probabilities.size
                xp.add.at(self.W_out.grad, candidates, gradient[:, :, None] * hidden[:, None, :])
                dhidden = xp.sum(gradient[:, :, None] * self.W_out.data[candidates], axis=1)
            if self.architecture == "cbow":
                width = source.shape[1]
                xp.add.at(self.W_in.grad, source, dhidden[:, None, :] / width)
            else:
                xp.add.at(self.W_in.grad, source, dhidden)
        return None
