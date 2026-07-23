"""Word2vec models with full-softmax and negative-sampling objectives."""

from __future__ import annotations

from collections.abc import Sequence
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
        loss_reduction: str = "mean",
        sampler: UnigramSampler | None = None,
        backend: Backend | str | None = None,
    ) -> None:
        resolved = resolve_backend(backend)
        super().__init__(resolved)
        if architecture not in {"cbow", "skipgram"}:
            raise ValueError("architecture must be 'cbow' or 'skipgram'")
        if objective not in {"full_softmax", "negative_sampling"}:
            raise ValueError("objective must be 'full_softmax' or 'negative_sampling'")
        if loss_reduction not in {"mean", "sum"}:
            raise ValueError("loss_reduction must be 'mean' or 'sum'")
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.architecture = architecture
        self.objective = objective
        self.negative_samples = negative_samples
        self.loss_reduction = loss_reduction
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
        self._gradient_scale = 1

    @property
    def word_vectors(self):
        return self.W_in.data

    def forward_manual(
        self,
        xs: Tensor,
        ts: Tensor,
        *,
        negative_candidates: Sequence[Any] | None = None,
    ) -> Tensor:
        xp = self.backend.xp
        indices = xs.data.astype(xp.int64, copy=False)
        targets = ts.data.astype(xp.int64, copy=False)
        self.cache = []
        self._gradient_scale = 1
        if self.architecture == "cbow":
            hidden = self.W_in.data[indices].mean(axis=1)
            fixed = None if negative_candidates is None else negative_candidates[0]
            loss = self._objective(hidden, targets.reshape(-1), indices, fixed_candidates=fixed)
        else:
            centers = indices.reshape(-1)
            context_targets = targets if targets.ndim == 2 else targets[:, None]
            context_width = context_targets.shape[1]
            flat_centers = xp.repeat(centers, context_width)
            flat_targets = context_targets.reshape(-1)
            fixed = self._flatten_negative_candidates(
                negative_candidates,
                batch_size=len(centers),
                context_width=context_width,
            )
            loss = self._objective(
                self.W_in.data[flat_centers], flat_targets, flat_centers,
                fixed_candidates=fixed,
            )
            # The former per-context implementation accumulated one gradient
            # contribution per context. Preserve that optimizer behavior while
            # issuing the objective and sampler kernels only once per batch.
            self._gradient_scale = context_width
            if self.loss_reduction == "sum":
                loss = loss * context_width
        return Tensor(xp.asarray(loss, dtype=self.backend.float_dtype), backend=self.backend)

    def _flatten_negative_candidates(
        self,
        candidates: Sequence[Any] | None,
        *,
        batch_size: int,
        context_width: int,
    ):
        if candidates is None:
            return None
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) != context_width:
            raise ValueError("negative candidate count must match the context width")
        xp = self.backend.xp
        return xp.stack(candidates, axis=1).reshape(batch_size * context_width, -1)

    def _objective(self, hidden, labels, source, *, fixed_candidates=None):
        xp = self.backend.xp
        if self.objective == "full_softmax":
            scores = hidden @ self.W_out.data.T
            scores -= scores.max(axis=1, keepdims=True)
            probabilities = xp.exp(scores)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            loss = -xp.log(probabilities[xp.arange(len(labels)), labels] + 1e-7).mean()
            self.cache.append((hidden, labels, source, probabilities))
            return loss
        if self.sampler is None:
            raise RuntimeError("negative-sampling objective requires a sampler")
        negatives = (
            fixed_candidates
            if fixed_candidates is not None
            else self.sampler.sample(labels, sample_size=self.negative_samples)
        )
        candidates = xp.concatenate((labels[:, None], negatives), axis=1)
        scores = xp.sum(hidden[:, None, :] * self.W_out.data[candidates], axis=2)
        targets = xp.zeros_like(scores)
        targets[:, 0] = 1
        probabilities = 1 / (1 + xp.exp(-scores))
        terms = -(targets * xp.log(probabilities + 1e-7) + (1 - targets) * xp.log(1 - probabilities + 1e-7))
        loss = terms.mean() if self.loss_reduction == "mean" else terms.sum(axis=1).mean()
        self.cache.append((hidden, candidates, source, (probabilities, targets)))
        return loss

    def last_negative_candidates(self) -> list[Any] | None:
        """Return copies of the negatives drawn by the most recent forward pass."""
        if self.objective != "negative_sampling":
            return None
        return [labels_or_candidates[:, 1:].copy() for _, labels_or_candidates, _, _ in self.cache]

    def backward_manual(self, dout: Tensor | None = None) -> None:
        xp = self.backend.xp
        self.W_in.grad[...] = 0
        self.W_out.grad[...] = 0
        scale = (1.0 if dout is None else float(dout.data)) * self._gradient_scale
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
                denominator = probabilities.size if self.loss_reduction == "mean" else probabilities.shape[0]
                gradient = (probabilities - targets) * scale / denominator
                xp.add.at(self.W_out.grad, candidates, gradient[:, :, None] * hidden[:, None, :])
                dhidden = xp.sum(gradient[:, :, None] * self.W_out.data[candidates], axis=1)
            if self.architecture == "cbow":
                width = source.shape[1]
                xp.add.at(self.W_in.grad, source, dhidden[:, None, :] / width)
            else:
                xp.add.at(self.W_in.grad, source, dhidden)
        return None
