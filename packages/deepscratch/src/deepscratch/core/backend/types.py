from dataclasses import dataclass
from typing import Any, Literal, Protocol

type Array = Any
type DType = Any
type Device = str
type BackendName = Literal["numpy", "cupy"]
type FloatDTypeName = Literal["float32", "float64"]


class DeviceMismatchError(RuntimeError):
    pass


class RandomModule(Protocol):
    def seed(self, seed: int) -> None: ...

    def rand(self, *shape: int) -> Array: ...

    def randn(self, *shape: int) -> Array: ...

    def randint(
        self,
        low: int,
        high: int | None = None,
        size: Any | None = None,
    ) -> Array: ...


class ArrayModule(Protocol):
    random: RandomModule

    float32: DType
    float64: DType
    int64: DType
    bool_: DType

    def asarray(self, obj: Any, dtype: DType | None = None) -> Array: ...

    def array(self, obj: Any, dtype: DType | None = None) -> Array: ...


class HasDevice(Protocol):
    @property
    def device(self) -> str: ...


type DeviceTarget = str | HasDevice


@dataclass(frozen=True)
class BackendConfig:
    device: str = "cpu"
    dtype: FloatDTypeName = "float64"
    seed: int | None = None
    profile: bool = False
