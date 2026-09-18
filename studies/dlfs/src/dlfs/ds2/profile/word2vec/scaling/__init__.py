"""Construct and render implemented Word2Vec vocabulary-size scaling."""

from __future__ import annotations

from .constants import (
    CONDITIONS,
    CONTEXT_WIDTH,
    DEFAULT_CPU_VOCAB_SIZES,
    DEFAULT_RESULTS,
    DEFAULT_VOCAB_SIZES,
    EMBEDDING_SIZE,
    NEGATIVE_SAMPLES,
    default_vocabulary_sizes,
)
from .crossover import (
    summarize_crossovers,
)
from .measurement import (
    measure_scaling_condition,
)
from .render import (
    render_individual_scaling,
    render_scaling,
)
from .runner import (
    run,
)
from .workload import (
    ScalingWorkload,
    synthetic_scaling_batches,
)

__all__ = [
    "CONDITIONS",
    "CONTEXT_WIDTH",
    "DEFAULT_CPU_VOCAB_SIZES",
    "DEFAULT_RESULTS",
    "DEFAULT_VOCAB_SIZES",
    "EMBEDDING_SIZE",
    "NEGATIVE_SAMPLES",
    "ScalingWorkload",
    "default_vocabulary_sizes",
    "measure_scaling_condition",
    "render_individual_scaling",
    "render_scaling",
    "run",
    "summarize_crossovers",
    "synthetic_scaling_batches",
]
