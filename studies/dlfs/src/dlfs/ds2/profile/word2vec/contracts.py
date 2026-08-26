"""Typed condition declarations for DS2 Word2Vec profiling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Word2VecCondition:
    subject_variant: Literal["original", "implemented"]
    model: Literal["cbow", "skipgram"]
    objective: Literal["full_softmax", "negative_sampling", "fused_negative_sampling"]
    input_representation: Literal["embedding", "one_hot"] = "embedding"

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> Word2VecCondition:
        condition = cls(
            subject_variant=str(values.get("subject_variant", "implemented")),  # type: ignore[arg-type]
            model=str(values.get("model", "")),  # type: ignore[arg-type]
            objective=str(values.get("objective", "")),  # type: ignore[arg-type]
            input_representation=str(values.get("input_representation", "embedding")),  # type: ignore[arg-type]
        )
        if condition.subject_variant not in {"original", "implemented"}:
            raise ValueError("unknown profile subject variant")
        if condition.model not in {"cbow", "skipgram"}:
            raise ValueError("unknown Word2Vec profile model")
        if condition.objective not in {
            "full_softmax",
            "negative_sampling",
            "fused_negative_sampling",
        }:
            raise ValueError("unknown Word2Vec profile objective")
        if condition.input_representation not in {"embedding", "one_hot"}:
            raise ValueError("unknown Word2Vec profile input representation")
        if (
            condition.subject_variant == "original"
            and condition.objective == "fused_negative_sampling"
        ):
            raise ValueError("the original implementation has no fused objective")
        if (
            condition.input_representation == "one_hot"
            and condition.objective != "full_softmax"
        ):
            raise ValueError("one-hot input requires full softmax")
        return condition

    def legacy_id(self) -> str:
        objective = {
            "full_softmax": (
                "onehot-fs" if self.input_representation == "one_hot" else "fs"
            ),
            "negative_sampling": "ns",
            "fused_negative_sampling": "fused-ns",
        }[self.objective]
        return f"{self.subject_variant}-{self.model}-{objective}"
