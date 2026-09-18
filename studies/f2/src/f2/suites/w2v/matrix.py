"""Canonical Word2Vec matrix identities and resource estimates."""

from dataclasses import dataclass

CANONICAL_SEEDS = (1, 7, 19)


@dataclass(frozen=True)
class MatrixCondition:
    atomic_run_id: str
    classification: str
    training_tokens: int
    embedding_dimension: int
    epochs: int
    device: str = "cpu"
    threads: int = 1

    @property
    def estimated_token_updates(self) -> int:
        return self.training_tokens * self.epochs


def planned_slot_id(execution_plan_id: str, atomic_run_id: str, seed: int) -> str:
    """Return the immutable identity shared by catalog, runtime, and reports."""
    return f"{execution_plan_id}-{atomic_run_id}-s{seed}"


def w2v1_conditions() -> tuple[MatrixCondition, ...]:
    return tuple(
        MatrixCondition(
            atomic_run_id=f"d{dimension}-w{tokens // 1_000_000}m",
            classification="reconstruction",
            training_tokens=tokens,
            embedding_dimension=dimension,
            epochs=3,
        )
        for dimension in (50, 100, 300, 600)
        for tokens in (24, 49, 98, 196, 391, 783)
        for tokens in (tokens * 1_000_000,)
    )


def w2v2_conditions() -> tuple[MatrixCondition, ...]:
    return tuple(
        MatrixCondition(
            atomic_run_id=atomic_run_id,
            classification="reconstruction",
            training_tokens=1_000_000_000,
            embedding_dimension=300,
            epochs=3,
        )
        for atomic_run_id in (
            "neg5-no-subsampling",
            "neg15-no-subsampling",
            "hs-no-subsampling",
            "neg5-subsampling",
            "neg15-subsampling",
            "hs-subsampling",
        )
    )


def canonical_slots(
    execution_plan_id: str, conditions: tuple[MatrixCondition, ...]
) -> tuple[dict[str, object], ...]:
    """Expand conditions into deterministic seed repetitions."""
    return tuple(
        {
            "planned_run_slot_id": planned_slot_id(
                execution_plan_id, condition.atomic_run_id, seed
            ),
            "atomic_run_id": condition.atomic_run_id,
            "seed": seed,
            "classification": condition.classification,
            "training_tokens": condition.training_tokens,
            "embedding_dimension": condition.embedding_dimension,
            "epochs": condition.epochs,
            "estimated_token_updates": condition.estimated_token_updates,
            "device": condition.device,
            "threads": condition.threads,
            "requires_approval": True,
        }
        for condition in conditions
        for seed in CANONICAL_SEEDS
    )


__all__ = [
    "CANONICAL_SEEDS",
    "MatrixCondition",
    "canonical_slots",
    "planned_slot_id",
    "w2v1_conditions",
    "w2v2_conditions",
]
