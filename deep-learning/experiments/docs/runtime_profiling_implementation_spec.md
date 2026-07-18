# Runtime Monitoring and Optional Profiling Implementation Specification

## 1. 문서 목적

NumPy/CuPy 백엔드를 공통으로 사용하는 학습 코드에 다음 기능을 추가한다.

1. 모든 학습 실행에서 저비용 공통 성능 지표 수집
2. 옵션 활성화 시에만 상세 프로파일링 수행
3. NumPy와 CuPy의 측정 차이를 Trainer 외부로 숨김
4. 측정 결과를 MLflow 또는 일반 로거에 바로 전달할 수 있는 평탄한 딕셔너리 형태로 제공
5. 기존 학습 결과와 실행 흐름을 변경하지 않음

이 문서는 Codex가 추가 설계 질문 없이 구현할 수 있도록 파일 구조, 인터페이스, 동작 규칙, 테스트 기준을 정의한다.

---

## 2. 구현 범위

### 2.1 항상 활성화되는 공통 측정

다음 항목은 상세 프로파일링 활성화 여부와 무관하게 수집한다.

- 전체 학습 시간
- 에폭별 실행 시간
- 에폭별 처리량
- 실행 시작/종료 시 CPU 메모리
- 실행 시작/종료 시 GPU 메모리
- 학습 중 샘플링된 CPU/GPU 최대 메모리
- 모델 파라미터 수
- 모델 파라미터 메모리
- 그래디언트 메모리
- 옵티마이저 상태 메모리

### 2.2 옵션 활성화 시 상세 측정

상세 프로파일링 옵션이 활성화된 경우에만 지정된 global step 범위에서 다음 항목을 수집한다.

- 전체 train step 시간
- forward 시간
- backward 시간
- gradient clipping 시간
- optimizer update 시간
- 상세 측정 구간별 메모리 스냅샷
- CuPy 사용 시 NVTX range
- 선택적으로 Python `cProfile`
- 선택적으로 Python `tracemalloc`

### 2.3 구현하지 않는 항목

이번 작업에서는 다음을 자동 실행하지 않는다.

- Nsight Systems 프로세스 실행
- Nsight Compute 프로세스 실행
- 외부 `nvidia-smi` 샘플러 프로세스 실행
- 커널별 occupancy 또는 memory throughput 분석
- 정확한 GPU 프로세스 peak memory 조회를 위한 NVML 통합

단, CuPy 상세 프로파일링 시 NVTX range를 남겨 사용자가 외부에서 Nsight Systems를 실행할 수 있게 한다.

---

## 3. 핵심 설계 원칙

### 3.1 Trainer와 백엔드별 구현 분리

Trainer는 NumPy 또는 CuPy 여부를 직접 검사하지 않는다.

Trainer는 다음 추상 인터페이스만 사용한다.

- `synchronize()`
- `memory_stats()`
- `range()`

백엔드별 차이는 profiler adapter가 처리한다.

### 3.2 공통 측정과 상세 프로파일링 분리

구성 요소를 다음처럼 분리한다.

```text
RuntimeMonitor
- 항상 사용하는 저비용 시간/메모리 수집

ProfilingController
- 현재 global step이 상세 측정 대상인지 판단

DetailProfiler
- cProfile, tracemalloc, NVTX range 관리

BackendProfiler
- NumPy/CuPy 동기화 및 메모리 조회 추상화
```

### 3.3 CuPy 비동기 실행 고려

CuPy 연산 시간은 측정 전후에 현재 CUDA stream을 동기화해야 실제 GPU 실행 시간을 포함한다.

단, 매 train step마다 동기화하면 정상 학습 처리량이 왜곡된다.

따라서 다음 규칙을 적용한다.

- 에폭 시간: 에폭 시작/종료 시에만 동기화
- 전체 학습 시간: 학습 시작/종료 시에만 동기화
- 상세 step 시간: 상세 프로파일링 대상으로 선택된 step에서만 구간별 동기화
- 일반 step: 동기화하지 않음

### 3.4 측정 실패가 학습을 중단하지 않도록 처리

프로파일링 관련 선택 기능이 지원되지 않더라도 학습은 계속되어야 한다.

예:

- `psutil` 세부 필드 미지원
- CuPy 미설치
- NVTX import 실패
- `tracemalloc` 비활성화 또는 중복 시작

필수 공통 기능 자체가 실패한 경우에는 명확한 예외를 발생시킨다. 선택적 상세 기능 실패는 경고 후 비활성화한다.

---

## 4. 권장 파일 구조

현재 프로젝트 구조에 맞춰 다음 파일을 추가한다.

```text
src/mlprosection/
├── profiling/
│   ├── __init__.py
│   ├── config.py
│   ├── backend.py
│   ├── monitor.py
│   ├── controller.py
│   ├── detail.py
│   └── utils.py
└── trainer/
    ├── base.py
    └── forward_trainer.py

tests/
├── profiling/
│   ├── test_config.py
│   ├── test_controller.py
│   ├── test_monitor.py
│   ├── test_numpy_backend_profiler.py
│   └── test_detail_profiler.py
└── trainer/
    └── test_trainer_profiling.py
```

CuPy 테스트는 CuPy가 설치되고 CUDA 장치를 사용할 수 있을 때만 실행한다.

---

## 5. 설정 객체

파일: `src/mlprosection/profiling/config.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfilingConfig:
    enabled: bool = False

    start_step: int = 0
    num_steps: int = 10

    profile_python: bool = False
    profile_memory: bool = False
    profile_gpu_ranges: bool = False

    collect_common_metrics: bool = True
    collect_epoch_metrics: bool = True
    collect_memory_metrics: bool = True
    collect_model_metrics: bool = True

    sample_memory_every_n_steps: int = 1
```

### 5.1 검증 규칙

`__post_init__()`에서 다음을 검증한다.

- `start_step >= 0`
- `num_steps >= 0`
- `sample_memory_every_n_steps >= 1`
- `enabled=False`여도 공통 측정 옵션은 독립적으로 작동
- `num_steps == 0`이면 상세 step 측정 없음

`frozen=True`를 유지한다.

---

## 6. 백엔드 프로파일러 인터페이스

파일: `src/mlprosection/profiling/backend.py`

### 6.1 Protocol

```python
from __future__ import annotations

from typing import Protocol


MetricValue = int | float


class BackendProfiler(Protocol):
    @property
    def name(self) -> str:
        ...

    def synchronize(self) -> None:
        ...

    def memory_stats(self) -> dict[str, MetricValue]:
        ...
```

### 6.2 NumPy 구현

```python
class NumPyBackendProfiler:
    name = "numpy"
```

동작:

- `synchronize()`는 아무 작업도 하지 않는다.
- `psutil.Process(os.getpid())`를 생성자에서 한 번만 생성한다.
- `memory_stats()`는 최소 다음 키를 반환한다.

```text
cpu.rss_bytes
cpu.vms_bytes
```

`memory_full_info()`에서 `uss`, `pss`, `swap`을 지원하면 다음 키도 추가한다.

```text
cpu.uss_bytes
cpu.pss_bytes
cpu.swap_bytes
```

지원하지 않는 필드는 키 자체를 생략한다. `None`을 반환하지 않는다.

### 6.3 CuPy 구현

```python
class CuPyBackendProfiler:
    name = "cupy"
```

동작:

- `synchronize()`는 `cp.cuda.get_current_stream().synchronize()`를 호출한다.
- 생성자에서 다음 객체를 캐시한다.
  - 현재 프로세스의 `psutil.Process`
  - `cp.get_default_memory_pool()`
  - `cp.get_default_pinned_memory_pool()`
- `memory_stats()`는 CPU 메모리와 함께 최소 다음 키를 반환한다.

```text
gpu.pool_used_bytes
gpu.pool_reserved_bytes
gpu.pinned_free_blocks
```

의미:

- `gpu.pool_used_bytes`: 현재 살아 있는 CuPy 배열이 사용하는 메모리
- `gpu.pool_reserved_bytes`: CuPy 메모리 풀이 CUDA에서 확보한 전체 메모리
- `gpu.pinned_free_blocks`: pinned memory pool의 재사용 가능 block 수

### 6.4 팩토리

```python
def create_backend_profiler(backend) -> BackendProfiler:
    ...
```

백엔드 판별은 프로젝트의 기존 `Backend` API를 사용한다.

권장 우선순위:

1. `backend.name`
2. `backend.xp.__name__`

지원하지 않는 백엔드는 다음 예외를 발생시킨다.

```python
ValueError(f"Unsupported backend for profiling: {backend_name}")
```

Trainer 내부에서 `cupy`를 직접 import하거나 비교하지 않는다.

---

## 7. 프로파일링 대상 step 제어

파일: `src/mlprosection/profiling/controller.py`

```python
from __future__ import annotations

from .config import ProfilingConfig


class ProfilingController:
    def __init__(self, config: ProfilingConfig) -> None:
        self.config = config

    def should_profile(self, global_step: int) -> bool:
        if not self.config.enabled:
            return False

        if self.config.num_steps == 0:
            return False

        start = self.config.start_step
        end = start + self.config.num_steps
        return start <= global_step < end

    def should_sample_memory(self, global_step: int) -> bool:
        if not self.config.collect_memory_metrics:
            return False

        interval = self.config.sample_memory_every_n_steps
        return global_step % interval == 0
```

경계 규칙:

- `start_step` 포함
- `start_step + num_steps` 제외

예:

```text
start_step=10
num_steps=3

profile 대상: 10, 11, 12
profile 비대상: 9, 13
```

---

## 8. RuntimeMonitor

파일: `src/mlprosection/profiling/monitor.py`

### 8.1 책임

- 시간 측정값 누적
- 메모리 스냅샷 저장
- 샘플링된 peak memory 갱신
- 최종 평탄화된 metric dictionary 생성
- 외부 로거에 종속되지 않음

### 8.2 저장 구조

```python
class RuntimeMonitor:
    def __init__(self, backend_profiler: BackendProfiler) -> None:
        self.backend_profiler = backend_profiler
        self.timings_ms: dict[str, list[float]] = {}
        self.memory_snapshots: dict[str, dict[str, MetricValue]] = {}
        self.memory_peaks: dict[str, MetricValue] = {}
        self.scalar_metrics: dict[str, MetricValue] = {}
```

### 8.3 시간 측정 context manager

```python
@contextmanager
def timer(
    self,
    name: str,
    *,
    synchronize: bool = False,
):
    ...
```

규칙:

1. `synchronize=True`이면 시작 직전에 동기화
2. `perf_counter_ns()`로 시작 시간 기록
3. context 종료 시 `synchronize=True`이면 종료 직전에 동기화
4. 경과 시간을 millisecond로 변환
5. `timings_ms[name]` 리스트에 append
6. context 내부에서 예외가 발생해도 측정값은 기록한 뒤 원래 예외를 다시 발생

주의:

- 공통 에폭/전체 시간은 호출부가 시작/종료 동기화를 명시하거나 `synchronize=True`를 사용한다.
- 일반 train step마다 사용하지 않는다.

### 8.4 메모리 스냅샷

```python
def snapshot_memory(self, name: str, *, synchronize: bool = False) -> None:
    ...
```

동작:

- 필요 시 동기화
- `backend_profiler.memory_stats()` 결과 저장
- 동일한 이름으로 다시 호출하면 최신 값으로 덮어씀

권장 snapshot 이름:

```text
run.start
train.start
train.end
run.end
epoch.{epoch}.start
epoch.{epoch}.end
profile.step.{global_step}.before
profile.step.{global_step}.after
```

### 8.5 peak memory 샘플링

```python
def update_memory_peaks(self) -> None:
    ...
```

동작:

- 동기화하지 않는다.
- 현재 `memory_stats()`를 읽는다.
- 각 key별 최대값 갱신

메트릭 이름에는 최종적으로 다음 prefix를 사용한다.

```text
memory.peak_sampled.<backend metric key>
```

예:

```text
memory.peak_sampled.cpu.rss_bytes
memory.peak_sampled.gpu.pool_used_bytes
memory.peak_sampled.gpu.pool_reserved_bytes
```

이는 정확한 순간 peak가 아니라 step 경계에서 샘플링한 최대값임을 이름으로 명시한다.

### 8.6 scalar metric 기록

```python
def set_metric(self, name: str, value: MetricValue) -> None:
    self.scalar_metrics[name] = value
```

다음 계산 결과를 저장할 때 사용한다.

- epoch throughput
- parameter count
- parameter bytes
- gradient bytes
- optimizer state bytes

### 8.7 시간 통계 요약

파일: `src/mlprosection/profiling/utils.py`

```python
def summarize_values(values: list[float]) -> dict[str, float]:
    ...
```

반환 키:

```text
count
mean
std
min
max
p50
p95
```

규칙:

- 빈 리스트: 빈 dict
- 원소 1개: `std=0.0`
- percentile은 선형 보간 또는 Python 표준 라이브러리 기반으로 일관되게 구현
- 외부 NumPy 의존성 없이 동작하게 작성

### 8.8 최종 metric dictionary

```python
def metrics(self) -> dict[str, MetricValue]:
    ...
```

반환 예:

```text
runtime.train_total.mean_ms
runtime.train_total.p95_ms
runtime.epoch.mean_ms
runtime.epoch.std_ms
runtime.profile.forward.mean_ms
runtime.profile.backward.mean_ms

memory.run.start.cpu.rss_bytes
memory.train.end.gpu.pool_used_bytes
memory.peak_sampled.cpu.rss_bytes

throughput.epoch.0.samples_per_s
model.parameter_count
model.parameter_bytes
model.gradient_bytes
optimizer.state_bytes
```

시간 metric 생성 규칙:

```text
runtime.<timer name>.<summary key>_ms
```

단, `count`는 단위 suffix를 붙이지 않는다.

예:

```text
runtime.profile.forward.count
runtime.profile.forward.mean_ms
runtime.profile.forward.p95_ms
```

---

## 9. DetailProfiler

파일: `src/mlprosection/profiling/detail.py`

### 9.1 책임

- 상세 프로파일링이 비활성화되어 있을 때 no-op context 제공
- 선택적으로 NVTX range 적용
- 선택적으로 `cProfile` 시작/종료 및 결과 저장
- 선택적으로 `tracemalloc` 시작/종료 및 peak 기록

### 9.2 생성자

```python
class DetailProfiler:
    def __init__(
        self,
        config: ProfilingConfig,
        backend_profiler: BackendProfiler,
        monitor: RuntimeMonitor,
    ) -> None:
        ...
```

### 9.3 상세 구간 context manager

```python
@contextmanager
def section(self, name: str, *, enabled: bool):
    ...
```

동작:

- `enabled=False`: 즉시 no-op
- `enabled=True`:
  - `monitor.timer(f"profile.{name}", synchronize=True)` 적용
  - CuPy backend이고 `profile_gpu_ranges=True`이면 `cupyx.profiler.time_range(name)` 적용
  - 두 context는 `ExitStack`으로 중첩

NumPy backend에서 `profile_gpu_ranges=True`여도 NVTX는 적용하지 않는다.

### 9.4 cProfile

다음 메서드를 제공한다.

```python
def start_run(self) -> None:
    ...

def stop_run(self) -> None:
    ...

def dump_artifacts(self, output_dir: str | Path) -> list[Path]:
    ...
```

규칙:

- `profile_python=False`이면 no-op
- `start_run()`에서 `cProfile.Profile().enable()`
- `stop_run()`에서 disable
- `dump_artifacts()`에서 다음 파일 생성

```text
python_profile.prof
python_profile.txt
```

`python_profile.txt`는 `pstats.Stats`를 사용하고 `cumulative` 기준 상위 100개를 기록한다.

파일 생성 실패 시 원인을 포함한 warning을 발생시키고 학습 결과는 유지한다.

### 9.5 tracemalloc

규칙:

- `profile_memory=False`이면 사용하지 않는다.
- `start_run()`에서 `tracemalloc.start()`
- 이미 시작된 경우 기존 추적기를 재사용하되, 이 DetailProfiler가 소유하지 않은 추적기는 종료하지 않는다.
- `stop_run()`에서 `get_traced_memory()` 호출
- 다음 metric을 monitor에 기록

```text
memory.python_traced.current_bytes
memory.python_traced.peak_bytes
```

- 자신이 시작한 경우에만 `tracemalloc.stop()` 호출

---

## 10. 모델 및 옵티마이저 메모리 계산

파일 위치는 `monitor.py` 또는 `utils.py` 중 기존 코드 구조에 맞춰 결정한다.

다음 함수들을 구현한다.

```python
def count_parameter_elements(model) -> int:
    ...


def count_parameter_bytes(model) -> int:
    ...


def count_gradient_bytes(model) -> int:
    ...


def count_optimizer_state_bytes(optimizer) -> int:
    ...
```

### 10.1 파라미터 수

- `model.parameters()` iterator 사용
- 각 parameter의 `data.size` 합산
- 중복 parameter 객체는 `id(parameter)` 기준으로 한 번만 집계

### 10.2 파라미터 메모리

- 각 parameter의 `data.nbytes` 합산
- 중복 제거

### 10.3 그래디언트 메모리

- `parameter.grad is not None`인 경우 `grad.nbytes` 합산
- grad가 Tensor일 수도 있고 raw array일 수도 있으므로 기존 프로젝트 타입에 맞춰 처리
- 필요 시 `grad.data.nbytes` fallback 허용

### 10.4 옵티마이저 상태 메모리

옵티마이저 인스턴스의 속성을 재귀 순회한다.

집계 대상:

- NumPy ndarray
- CuPy ndarray
- Tensor 내부의 `.data`
- dict/list/tuple/set 내부 배열

제외 대상:

- model parameter 원본 배열
- gradient 원본 배열
- scalar
- callable
- module, class

중복 배열은 `id(array)` 기준으로 한 번만 집계한다.

재귀 순환 방지를 위해 방문 객체 id 집합을 사용한다.

이 함수는 optimizer 구현별 `m`, `v`, `h` 등 state 배열을 자동 집계해야 한다.

---

## 11. Trainer 통합

대상 파일:

- `src/mlprosection/trainer/base.py`
- 필요 시 `src/mlprosection/trainer/forward_trainer.py`

### 11.1 생성자 변경

기존 Trainer 생성자에 다음 인자를 추가한다.

```python
profiling_config: ProfilingConfig | None = None
```

기본값:

```python
self.profiling_config = profiling_config or ProfilingConfig()
```

다음 객체를 초기화한다.

```python
self.backend_profiler = create_backend_profiler(self.backend)
self.runtime_monitor = RuntimeMonitor(self.backend_profiler)
self.profiling_controller = ProfilingController(self.profiling_config)
self.detail_profiler = DetailProfiler(
    self.profiling_config,
    self.backend_profiler,
    self.runtime_monitor,
)
self.global_step = 0
```

기존 외부 API와 호환성을 유지한다.

### 11.2 fit 시작 처리

`fit()` 시작 직전에:

1. `detail_profiler.start_run()`
2. `runtime_monitor.snapshot_memory("run.start", synchronize=True)`
3. `runtime_monitor.snapshot_memory("train.start")`
4. model metric 수집
5. 전체 학습 timer 시작

권장 구조:

```python
self.detail_profiler.start_run()

try:
    self.runtime_monitor.snapshot_memory("run.start", synchronize=True)
    self.runtime_monitor.snapshot_memory("train.start")

    self._record_model_metrics()

    with self.runtime_monitor.timer(
        "train_total",
        synchronize=True,
    ):
        ... existing fit loop ...
finally:
    self.runtime_monitor.snapshot_memory("train.end", synchronize=True)
    self.runtime_monitor.snapshot_memory("run.end")
    self._record_final_memory_metrics()
    self.detail_profiler.stop_run()
```

학습 중 예외가 발생해도 profiler 종료와 마지막 snapshot을 시도한 뒤 원래 예외를 다시 발생시킨다.

### 11.3 에폭 측정

각 에폭 실행을 다음처럼 감싼다.

```python
with self.runtime_monitor.timer("epoch", synchronize=True):
    epoch_result = self.run_epoch(...)
```

에폭별 개별 값도 저장한다.

```text
runtime.epoch.<epoch>.duration_ms
throughput.epoch.<epoch>.samples_per_s
```

`RuntimeMonitor.timer()`의 집계값과 별도로 scalar metric에 에폭별 duration을 넣는다.

에폭 시간 계산 시 validation 시간 포함 여부는 명확히 분리한다.

권장:

```text
runtime.epoch.<epoch>.train_duration_ms
runtime.epoch.<epoch>.eval_duration_ms
throughput.epoch.<epoch>.train_samples_per_s
```

기존 `run_epoch()`가 train만 수행한다면 우선 train 기준으로 구현한다.

### 11.4 run_epoch 내부

각 batch 시작 시:

```python
profile_this_step = self.profiling_controller.should_profile(
    self.global_step
)
```

메모리 샘플링:

```python
if self.profiling_controller.should_sample_memory(self.global_step):
    self.runtime_monitor.update_memory_peaks()
```

상세 측정 대상이면 step 전후 snapshot을 기록한다.

```python
if profile_this_step and self.profiling_config.profile_memory:
    self.runtime_monitor.snapshot_memory(
        f"profile.step.{self.global_step}.before",
        synchronize=True,
    )
```

step 호출:

```python
loss = self.step(
    batch_x,
    batch_t,
    profile=profile_this_step,
)
```

step 이후:

```python
if profile_this_step and self.profiling_config.profile_memory:
    self.runtime_monitor.snapshot_memory(
        f"profile.step.{self.global_step}.after",
        synchronize=True,
    )

self.global_step += 1
```

### 11.5 step 시그니처

기존 `step()`에 keyword-only 인자를 추가한다.

```python
def step(
    self,
    batch_x,
    batch_t,
    *,
    profile: bool = False,
):
    ...
```

기존 호출은 그대로 동작해야 한다.

### 11.6 step 내부 상세 구간

중복 구현 없이 하나의 step 함수에서 context를 선택적으로 사용한다.

```python
def step(self, batch_x, batch_t, *, profile: bool = False):
    self.optimizer.zero_grad()

    with self.detail_profiler.section("forward", enabled=profile):
        y = self.model(batch_x)
        loss = self.criterion(y, batch_t)

    with self.detail_profiler.section("backward", enabled=profile):
        loss.backward()

    if self.max_grad is not None:
        with self.detail_profiler.section(
            "gradient_clip",
            enabled=profile,
        ):
            clip_grads(self.model.parameters(), self.max_grad)

    with self.detail_profiler.section("optimizer_update", enabled=profile):
        self.optimizer.update()

    return loss
```

현재 코드에서 `optimizer.zero_grad()` 대신 다른 초기화 방식을 사용하면 기존 동작을 유지한다.

`zero_grad`도 상세 측정이 필요하면 `profile.zero_grad` 구간으로 추가할 수 있으나 필수는 아니다.

---

## 12. 외부 공개 API

파일: `src/mlprosection/profiling/__init__.py`

다음만 공개한다.

```python
from .config import ProfilingConfig
from .monitor import RuntimeMonitor

__all__ = [
    "ProfilingConfig",
    "RuntimeMonitor",
]
```

Trainer 사용자는 내부 adapter 클래스를 직접 알 필요가 없다.

### 12.1 사용 예

기본 실행:

```python
trainer = ForwardTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
)

trainer.fit(x_train, t_train, x_valid, t_valid)
metrics = trainer.profiling_metrics()
```

상세 프로파일링 실행:

```python
from mlprosection.profiling import ProfilingConfig


profiling_config = ProfilingConfig(
    enabled=True,
    start_step=10,
    num_steps=20,
    profile_python=True,
    profile_memory=True,
    profile_gpu_ranges=True,
    sample_memory_every_n_steps=1,
)

trainer = ForwardTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    profiling_config=profiling_config,
)

trainer.fit(x_train, t_train, x_valid, t_valid)
metrics = trainer.profiling_metrics()
artifacts = trainer.dump_profiling_artifacts("artifacts/profiling")
```

### 12.2 Trainer 공개 메서드

```python
def profiling_metrics(self) -> dict[str, int | float]:
    return self.runtime_monitor.metrics()


def dump_profiling_artifacts(
    self,
    output_dir: str | Path,
) -> list[Path]:
    return self.detail_profiler.dump_artifacts(output_dir)
```

상세 profiling이 비활성화된 경우 `dump_profiling_artifacts()`는 빈 리스트를 반환한다.

---

## 13. MLflow 연동 규칙

이번 구현에서 MLflow를 직접 import하지 않는다.

호출부에서 다음처럼 사용 가능해야 한다.

```python
metrics = trainer.profiling_metrics()
mlflow.log_metrics(metrics)

for artifact in trainer.dump_profiling_artifacts(
    "artifacts/profiling"
):
    mlflow.log_artifact(str(artifact), artifact_path="profiles")
```

### 13.1 metric key 제한

- key는 영문 소문자, 숫자, 점, 밑줄만 사용
- slash 사용 금지
- 모든 메모리 단위는 byte
- 모든 시간 단위는 key suffix로 명시
- 처리량은 `samples_per_s`

### 13.2 권장 metric schema

```text
runtime.train_total.mean_ms
runtime.epoch.mean_ms
runtime.epoch.std_ms
runtime.profile.forward.mean_ms
runtime.profile.backward.mean_ms
runtime.profile.optimizer_update.mean_ms

runtime.epoch.0.train_duration_ms
throughput.epoch.0.train_samples_per_s

memory.run.start.cpu.rss_bytes
memory.run.end.cpu.rss_bytes
memory.peak_sampled.cpu.rss_bytes
memory.peak_sampled.gpu.pool_used_bytes
memory.peak_sampled.gpu.pool_reserved_bytes
memory.python_traced.peak_bytes

model.parameter_count
model.parameter_bytes
model.gradient_bytes
optimizer.state_bytes
```

---

## 14. 테스트 요구사항

모든 테스트는 pytest로 작성한다.

### 14.1 ProfilingConfig

검증 항목:

- 기본값 생성
- 음수 `start_step` 거부
- 음수 `num_steps` 거부
- 0 이하 memory sampling interval 거부
- frozen dataclass 확인

### 14.2 ProfilingController

검증 항목:

- `enabled=False`이면 항상 False
- 시작 step 포함
- 종료 step 제외
- `num_steps=0`이면 항상 False
- memory sampling interval 동작

### 14.3 RuntimeMonitor timer

검증 항목:

- 시간 기록 생성
- 동일 이름 반복 시 리스트 누적
- context 내부 예외 발생 시에도 시간 기록
- 원래 예외 재발생
- `synchronize=True`일 때 mock profiler의 synchronize가 정확히 2회 호출
- `synchronize=False`일 때 호출되지 않음

### 14.4 RuntimeMonitor memory

검증 항목:

- snapshot 이름별 저장
- 동일 이름 덮어쓰기
- peak key별 최대값 갱신
- metrics 평탄화

### 14.5 NumPyBackendProfiler

검증 항목:

- `synchronize()` no-op
- `cpu.rss_bytes`와 `cpu.vms_bytes` 존재
- 모든 반환값이 int 또는 float

### 14.6 DetailProfiler

mock backend와 monitor를 사용한다.

검증 항목:

- `enabled=False` section no-op
- `enabled=True` timer 진입
- NumPy backend에서 NVTX 미호출
- CuPy backend mock에서 NVTX 호출
- `profile_python=False`일 때 artifact 없음
- `profile_python=True`일 때 `.prof`, `.txt` 생성
- tracemalloc metric 생성

### 14.7 Trainer 통합

작은 mock model/optimizer/criterion을 사용한다.

검증 항목:

- profiling config 미전달 시 기존 학습 동작 유지
- 공통 metric 생성
- 지정 범위의 step만 상세 profile
- global step이 에폭 간 이어짐
- 상세 profile 비활성 시 forward/backward timer 없음
- 상세 profile 활성 시 forward/backward/update timer 존재
- 학습 예외 발생 시 profiler stop 실행
- metric 수집이 loss 또는 parameter update 결과를 바꾸지 않음

### 14.8 CuPy 테스트

다음 조건을 만족할 때만 실행한다.

```python
pytest.importorskip("cupy")
```

추가로 CUDA device 접근 실패 시 skip한다.

검증 항목:

- stream synchronize 호출 가능
- memory pool key 반환
- 간단한 CuPy 연산을 상세 timer로 측정 가능

---

## 15. 성능 및 오버헤드 기준

### 15.1 상세 프로파일링 비활성

`ProfilingConfig(enabled=False)` 기준으로:

- 일반 train step마다 CUDA synchronize 호출 금지
- 일반 train step마다 `perf_counter_ns()` 구간 측정 금지
- memory sampling은 설정 주기에만 수행
- 기존 코드 대비 처리량 저하는 CPU 기준 5% 이내를 목표로 함
- GPU 기준 에폭 시간 측정 외 강제 동기화가 없어야 함

### 15.2 상세 프로파일링 활성

상세 측정 대상 step에서는 정확한 구간 시간 수집을 위해 동기화를 허용한다.

해당 결과는 정상 학습 처리량 비교에 사용하지 않는다.

공통 처리량 metric은 상세 프로파일링이 활성화된 run에서 왜곡될 수 있으므로 다음 tag 또는 metric을 함께 기록할 수 있게 한다.

```text
profiling.enabled = 1
profiling.profiled_step_count = <num_steps>
```

이는 `RuntimeMonitor.scalar_metrics`에 저장한다.

---

## 16. 오류 처리

### 16.1 선택 기능 실패

다음 기능 실패는 warning 후 비활성화한다.

- NVTX range import
- cProfile artifact dump
- tracemalloc 부가 metric
- `memory_full_info()`의 선택 필드

### 16.2 필수 기능 실패

다음은 명시적 예외를 발생시킨다.

- 지원하지 않는 backend
- 잘못된 ProfilingConfig
- RuntimeMonitor에 잘못된 metric value 전달

### 16.3 로그

프로젝트 기존 로거가 있으면 이를 사용한다. 없으면 표준 `logging.getLogger(__name__)`를 사용한다.

`print()` 사용 금지.

---

## 17. 코드 품질 요구사항

- Python 3.11 기준
- `from __future__ import annotations` 사용
- ruff 통과
- mypy strict를 최대한 만족
- public class/function에 docstring 작성
- line length 80 기준
- backend-specific import는 가능한 한 해당 구현 내부로 제한
- CuPy 미설치 환경에서 profiling package import가 실패하면 안 됨
- 순환 import 금지
- Trainer가 MLflow에 직접 의존하지 않음

---

## 18. 구현 순서

Codex는 다음 순서로 구현한다.

1. `profiling/config.py`
2. `profiling/controller.py`
3. `profiling/backend.py`
4. `profiling/utils.py`
5. `profiling/monitor.py`
6. `profiling/detail.py`
7. `profiling/__init__.py`
8. Trainer 생성자 통합
9. fit/run_epoch/step 통합
10. model/optimizer memory 계산
11. 공개 metric/artifact 메서드 추가
12. 단위 테스트 작성
13. 기존 trainer 테스트 실행
14. 전체 ruff/pytest 실행

---

## 19. 완료 조건

다음을 모두 만족해야 작업 완료로 본다.

- NumPy 학습에서 공통 metric이 생성됨
- CuPy 설치 없이 package import 가능
- CuPy 환경에서 에폭 시간 측정이 실제 GPU 작업을 포함함
- 상세 프로파일링이 지정 step에서만 실행됨
- 상세 프로파일링 비활성 시 구간별 강제 동기화가 없음
- 학습 중 예외가 나도 profiler가 정리됨
- MLflow에 바로 넘길 수 있는 flat dict 반환
- cProfile artifact 생성 가능
- 기존 Trainer 공개 사용법이 깨지지 않음
- 신규 및 기존 pytest 통과
- ruff 통과

---

## 20. Codex 작업 지시문

아래 요구사항을 기준으로 구현한다.

```text
현재 NumPy/CuPy 공통 딥러닝 프레임워크의 Trainer에 런타임 측정과
선택적 상세 프로파일링 기능을 추가하라.

이 문서의 파일 구조, 공개 인터페이스, metric naming, 동기화 규칙,
오류 처리, 테스트 기준을 따른다.

중요 제약:
1. Trainer는 NumPy/CuPy를 직접 분기하지 않는다.
2. CuPy 일반 학습 step마다 synchronize하지 않는다.
3. 상세 측정 대상으로 선택된 step에서만 구간별 synchronize한다.
4. CuPy가 설치되지 않아도 package import가 가능해야 한다.
5. MLflow를 profiling 모듈 내부에서 직접 import하지 않는다.
6. 기존 Trainer 호출 방식과 학습 결과를 유지한다.
7. 구현 후 ruff와 pytest를 실행하고 실패를 수정한다.

구현 완료 후 다음을 보고하라.
- 변경 파일 목록
- 주요 설계 결정
- 생성되는 metric 목록
- 테스트 실행 결과
- 남은 제한사항
```
