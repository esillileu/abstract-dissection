from __future__ import annotations

from dataclasses import dataclass

from deepscratch.core import Tensor


@dataclass(frozen=True)
class Word2VecObjectiveBatch:
    target: Tensor
    candidates: Tensor | None = None
    replay_context: object = None
