# scratchdl/backend.py

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import ContextManager, Any
from .types import (
    ArrayModule,
    BackendName,
    DType,
    Array,
)


@dataclass
class Backend:
    xp: ArrayModule
    name: BackendName
    device: str
    float_dtype: DType
    int_dtype: DType
    bool_dtype: DType
    profile: bool = False
    _random_streams: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_cpu(self) -> bool:
        return self.device == "cpu"

    @property
    def is_gpu(self) -> bool:
        return self.device.startswith("cuda:")

    @property
    def dtype_name(self) -> str:
        if self.float_dtype == self.xp.float32:
            return "float32"
        if self.float_dtype == self.xp.float64:
            return "float64"
        return str(self.float_dtype)

    def asarray(self, obj: Any, dtype: DType | None = None) -> Array:
        return self.xp.asarray(obj, dtype=dtype)

    def asfloat(self, obj: Any) -> Array:
        return self.xp.asarray(obj, dtype=self.float_dtype)

    def asint(self, obj: Any) -> Array:
        return self.xp.asarray(obj, dtype=self.int_dtype)

    def to_numpy(self, array: Array) -> Array:
        if self.is_gpu:
            return array.get()
        return array

    def move_array_from(
        self, array: Array, source: Backend, dtype: DType | None = None
    ) -> Array:
        if source.device == self.device:
            if dtype is None:
                return array
            return self.xp.asarray(array, dtype=dtype)

        cpu_array = source.to_numpy(array)
        return self.xp.asarray(cpu_array, dtype=dtype)

    def scalar_to_float(self, scalar: Array) -> float:
        return float(self.to_numpy(scalar))

    def scalar_to_int(self, scalar: Array) -> int:
        return int(self.to_numpy(scalar))

    def seed(self, seed: int) -> None:
        self.xp.random.seed(seed)

    def configure_random_streams(self, seeds: dict[str, int]) -> None:
        """Create independent, persistent backend RNGs for experiment policy."""
        random_state = self.xp.random.RandomState
        self._random_streams = {
            name: random_state(int(seed)) for name, seed in seeds.items()
        }

    def random_stream(self, name: str):
        """Return a configured component RNG, falling back for legacy callers."""
        return self._random_streams.get(name, self.xp.random)

    def random_stream_states(self) -> dict[str, Any]:
        return {
            name: rng.get_state()
            for name, rng in self._random_streams.items()
            if hasattr(rng, "get_state")
        }

    def restore_random_stream_states(self, states: dict[str, Any]) -> None:
        for name, state in states.items():
            rng = self._random_streams.get(name)
            if rng is not None and hasattr(rng, "set_state"):
                rng.set_state(state)

    def synchronize(self) -> None:
        if self.is_gpu:
            self.xp.cuda.Stream.null.synchronize()

    def range(self, name: str) -> ContextManager[Any]:
        if self.is_gpu and self.profile:
            from cupyx.profiler import time_range

            return time_range(name)

        return nullcontext()

    def memory_info(self) -> dict[str, int] | None:
        if not self.is_gpu:
            return None

        mempool = self.xp.get_default_memory_pool()
        pinned_mempool = self.xp.get_default_pinned_memory_pool()

        return {
            "used_bytes": mempool.used_bytes(),
            "total_bytes": mempool.total_bytes(),
            "pinned_free_blocks": pinned_mempool.n_free_blocks(),
        }

    def clear_memory_pool(self) -> None:
        if not self.is_gpu:
            return

        self.xp.get_default_memory_pool().free_all_blocks()
        self.xp.get_default_pinned_memory_pool().free_all_blocks()
