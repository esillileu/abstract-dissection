from dataclasses import dataclass

from typing import Any, Literal, Protocol, TypeAlias

Array: TypeAlias = Any
DType: TypeAlias = Any
Device: TypeAlias = str
BackendName: TypeAlias = Literal["numpy", "cupy"]
FloatDTypeName: TypeAlias = Literal["float32", "float64"]


class DeviceMismatchError(RuntimeError):
    pass


class RandomModule(Protocol):
    def seed(self, seed: int) -> None:
        ...

    def rand(self, *shape: int) -> Array:
        ...

    def randn(self, *shape: int) -> Array:
        ...

    def randint(
        self,
        low: int,
        high: int | None = None,
        size: Any | None = None,
    ) -> Array:
        ...


class ArrayModule(Protocol):
    random: RandomModule

    float32: DType
    float64: DType
    int64: DType
    bool_: DType

    def asarray(self, obj: Any, dtype: DType | None = None) -> Array:
        ...

    def array(self, obj: Any, dtype: DType | None = None) -> Array:
        ...


class HasDevice(Protocol):
    @property
    def device(self) -> str:
        ...


DeviceTarget: TypeAlias = str | HasDevice


@dataclass(frozen=True)
class BackendConfig:
    device: str = "cpu"
    dtype: FloatDTypeName = "float64"
    seed: int | None = None
    profile: bool = False
