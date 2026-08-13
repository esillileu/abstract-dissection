# BetterRnnlm CUDA acceleration report

## 결론

Phase 1–3은 correctness와 단계별 최소 1% full-update 성능 gate를 통과했다.
Phase 2의 fused TimeLSTM과 Phase 3의 fused temporal softmax는 production
`src/mlprosection`으로 승격했다. CUDA float32에서만 fast path를 사용하며 CPU와
다른 dtype은 기존 vectorized 구현을 유지한다.

| 단계 | full update mean ± stdev (ms) | 직전 단계 대비 | baseline 대비 |
|---|---:|---:|---:|
| baseline | 81.688 ± 0.566 | — | 1.000× |
| Phase 1: sequence GEMM | 65.678 ± 3.180 | 19.60% 감소 | 1.244× |
| Phase 2: fused LSTM elementwise | 19.044 ± 0.351 | 71.00% 감소 | 4.289× |
| Phase 3: fused temporal softmax | 18.321 ± 0.059 | 3.80% 감소 | 4.459× |

모든 phase의 다섯 measurement window가 직전 단계보다 빨랐다. 실제 40-epoch
학습은 실행하지 않았다. epoch당 1,327 steady updates로 외삽한 Phase 3 시간은
약 24.31초/epoch, 0.270시간/40 epochs(약 16.2분)다. evaluation, checkpoint,
MLflow I/O는 포함하지 않는다.

## 환경과 측정 계약

- baseline commit: `8e29004bf155bb701c068ab9659cf2db9de88838`
- NVIDIA GeForce RTX 4060 Laptop GPU, NVIDIA driver 610.88
- CuPy 14.1.1, CUDA runtime 12.9, driver API 13.3
- Nsight Systems 2026.4.1, Python 3.11.14, WSL2, float32
- TimeLSTM: `N=20, T=35, D=H=650`
- PTB: `V=10000`, batch 20, BPTT 35, two 650-unit stateful LSTMs,
  dropout 0.5, tied affine, mean temporal softmax, clip 0.25, SGD 20
- 20 warmups 뒤 50 consecutive updates를 5 windows에서 측정

초기 측정에서 `Dropout`의 Python scalar가 float32 activation을 float64로
승격시키는 dtype 계약 오류를 발견했다. 이를 production에서 수정한 뒤 baseline과
모든 phase를 동일한 실제 float32 경로로 다시 측정했다. 이전 421/362 ms 결과는
float64 경로이므로 폐기했다.

## Phase별 변경

### Phase 1

TimeLSTM input projection을 timestep별 GEMM에서 `X_flat @ Wx` 한 번으로 바꾸고,
backward의 dWx, dWh, db, dX를 reverse loop 이후 sequence 단위로 집계했다.
Nsight NVTX에서 각 LSTM의 forward input GEMM은 `T→1`, backward non-recurrent
GEMM은 `3T→3`으로 감소했다.

| 단독 TimeLSTM | baseline (ms) | Phase 1 (ms) |
|---|---:|---:|
| forward | 13.208 | 14.306 |
| backward | 23.948 | 15.441 |

### Phase 2

CUDA float32에서 gate activation, cell/hidden update와 cache 기록을 timestep당
한 kernel로 결합했다. backward도 upstream/recurrent gradient와 cached state에서
dc_prev와 F/G/I/O dA를 한 kernel로 만든다. recurrent GEMM과 CPU fallback은
변경하지 않았다.

두 h/c ping-pong buffer와 dc ping-pong buffer만 사용한다. BetterRnnlm 20-update
VRAM 확인에서 CuPy pool은 약 497 MiB에서 안정화됐고 update 5/10/20 사이 증가가
없었다. Nsight에서 LSTM elementwise 비중은 Phase 1의 9.45%에서 Phase 2의
3.63%로 감소했다.

### Phase 3

CUDA float32 temporal softmax에서 row max, exp/sum normalization, loss와 전체
logits gradient 생성을 fused kernel로 처리했다. logits는 수정하지 않으며
stabilization, ignore-label, mean/sum reduction과 `cache=False` 계약을 유지한다.

동일 환경 production 재측정에서 objective CUDA-event 평균은 1.572 → 0.659 ms로
감소했다. Nsight에서 objective는
update당 28 kernels에서 5 kernels로 감소했다.

## 정확성 및 재현성

CPU와 CUDA에서 seed 1/7/23, 경계 shape와 `(20,35,650,650)` 대표 shape를 frozen
reference와 비교했다. output/final h/c, dx/dWx/dWh/db/dh, consecutive stateful
forward, reset/detach, `cache=False`, batch-size 변경을 포함한다.

- 관측 forward/state 결합오차 최대: `1.131e-6`; 선택 tolerance `3e-6`
- 관측 gradient 결합오차 최대: `1.532e-5`; 선택 tolerance `1e-4`
- hard ceiling: forward `1e-4`, gradient `1e-3`
- reference/production PTB 5-update lockstep: 통과, NaN/Inf 없음
- 동일 seed production 2회: loss 최대 절대차 `9.537e-7`, parameter/state tolerance 통과

재현성 artifact에는 tolerance 판정과 함께 loss bitwise 일치 여부도 별도 기록한다.
Phase 3 개발 중 발견한 RawKernel scalar ABI 타입 불일치와 shared-memory buffer
재사용 barrier 누락은 tolerance를 늘리지 않고 구현을 수정했다.

## Nsight 결과와 다음 병목

Nsight Systems 2026.4.1에서 CUDA kernel activity를 정상 수집했다. Phase 2
full update의 kernel time은 update당 대략 GEMM 11.17 ms, elementwise 4.41 ms,
reduction 0.67 ms였다. Phase 3 이후 측정상 clipping은 1.106 ms(6.5%), SGD는
0.971 ms(5.7%)로 다음 독립 후보지만 아직 변경하지 않았다.

Raw JSON/CSV/SQLite/Nsight 결과는 ignored `results/` 아래에 있고 이 보고서만
추적한다.

## Production 승격과 검증

- `nn/kernels/time_lstm.py`: production fused LSTM kernels
- `nn/layers/time.py`: CUDA float32 fast path 선택과 CPU/dtype fallback
- `nn/kernels/temporal_softmax.py`: production fused objective kernel
- `nn/objective/classification.py`: temporal softmax fast path 선택
- e05 `reference.py`, `phase1.py`, `phase2.py`, `phase3.py`: 과거 단계 재현 selector

승격된 `src` 경로로 CPU/CUDA correctness 검증을 다시 통과했다. WSL 재부팅으로
GPU가 복구된 뒤 baseline부터 Phase 3까지 같은 환경에서 공식 protocol을 다시
실행했으며 위 표는 이 production 재측정 결과다. 각 단계의 다섯 window 모두
직전 단계보다 빨라 performance gate도 통과했다.

```text
uv run --extra tracking pytest tests exp/ds2/tests -q  # 328 passed, 7 skipped
ruff check .                                           # passed
git diff --check                                       # passed
```
