# 시행 기록 스키마

목표는 원본 그래프를 같은 x축 좌표와 raw 값으로 재현하고, 분석 단계가 추가 통계를 계산할 수 있게 하는 것이다. 이 문서는 `GT01`–`GT08`, `GO01`–`GO02`가 남겨야 하는 데이터와 기록 시점을 고정한다.

MLflow metric 이름과 artifact layout의 구현 계약은 [`mlflow/`](mlflow/)에 둔다.

분석 단계는 이 문서의 raw record만 읽는다. smoothing, final/best, AUC, 평균, CI, 순위, paired difference는 여기서 기록하지 않는다.

## 저장 규칙

- 값은 지정된 update/epoch에서 계산한다.
- `updates.csv`, `evaluations.csv`, `timing_windows.csv`는 메모리 buffer에 쌓아 256 records마다, epoch 종료, checkpoint 직전, run 종료에 flush한다.
- buffer flush는 계산 step이나 값에 영향을 주지 않는다.
- MLflow에는 같은 값을 batch로 전송한다. artifact CSV가 완전한 history의 기준이다.

## 모든 run의 파일

```text
artifact/
  manifest.json
  updates.csv
  evaluations.csv
  timing_windows.csv         # GT 그룹만 사용
  checkpoints.csv
  observations/              # GO 그룹만 사용
```

### `manifest.json`

| 필드 | 값 |
| --- | --- |
| `group_id` | `GTnn` 또는 `GOnn` |
| `atomic_run_id` | 조건 ID |
| `master_seed` | trial seed |
| `model_seed`, `batch_order_seed` | 파생 seed |
| `dataset_id`, `dataset_checksum` | 데이터 식별·검증값 |
| `split_id`, `split_checksum` | subset/split 식별·검증값 |
| `resolved_config` | 모델·optimizer·sampler·budget 전체 |
| `loss_phase` | 항상 `post_update` |
| `loss_reduction` | 항상 `mean` |
| `evaluation_schedule` | 아래 그룹별 schedule 전체 |
| `timing_config` | timing window 경계, wall/device time 측정 여부와 단위 |

### `updates.csv`

모든 `GT` 학습 group은 매 optimizer update 뒤 한 행을 기록한다.

| 열 | 의미 |
| --- | --- |
| `update` | 완료된 optimizer update 수. 1부터 시작 |
| `epoch` | 해당 update가 속한 epoch. 1부터 시작 |
| `batch_size` | 실제 batch example 수 |
| `loss` | update **후** 동일 batch에서 다시 계산한 mean objective |
| `lr` | 해당 update에 적용한 learning rate |

MLflow mapping:

| CSV 열 | MLflow metric | MLflow step |
| --- | --- | --- |
| `loss` | `update/train/loss` | `update` |
| `lr` | `update/train/lr` | `update` |

### `evaluations.csv`

평가마다 split별로 한 행을 기록한다. `loss`와 `accuracy` 중 정의되지 않거나 측정하지 않은 값은 빈값이다.

| 열 | 의미 |
| --- | --- |
| `axis` | `update`, `epoch`, `terminal` 중 하나 |
| `axis_step` | 그래프 x축 값. update/epoch이면 해당 번호, terminal이면 마지막 update |
| `update` | 평가 직전 완료된 update 수 |
| `epoch` | 평가 시점 epoch |
| `evaluation_set_id` | 예: `mnist-train-first-300`, `mnist-test-full` |
| `split` | `train` 또는 `test` |
| `example_count` | 실제 평가 example 수 |
| `loss` | 해당 evaluation set의 mean objective |
| `accuracy` | `correct / example_count` |

MLflow mapping:

| `axis` | `split` | MLflow metric | MLflow step |
| --- | --- | --- | --- |
| `update` | train/test | `update/eval_{split}/accuracy` | `update` |
| `epoch` | train/test | `epoch/eval_{split}/accuracy` | `epoch` |
| `terminal` | train/test | `terminal/eval_{split}/accuracy` | 마지막 `update` |

`loss`를 평가에서 실제 측정하는 경우에는 같은 규칙으로 metric suffix를 `loss`로 바꾼다.

### `checkpoints.csv`

| 열 | 의미 |
| --- | --- |
| `update`, `epoch` | 저장 시점 |
| `kind` | `final`, `latest`, `selected`, `periodic` 중 하나 |
| `path` | artifact 내부 상대 경로 |
| `sha256` | checkpoint bytes hash |

`latest`와 metric-selected `selected`는 실제 보존된 한 세대만 기록한다. `periodic`은 명시적인 retention 정책이 있는 실행에서만 기록한다.

### `timing_windows.csv`

모든 `GT` 학습 group은 executor가 닫는 학습 timing window마다 한 행을 기록한다.
기본 window는 한 probe 종료 뒤 다음 probe 직전까지의 연속한 training update다. probe가
없는 group은 epoch 종료 또는 terminal에서 window를 닫는다.

| 열 | 의미 |
| --- | --- |
| `start_update`, `end_update` | window에 포함한 첫/마지막 완료 update. 둘 다 포함 범위 |
| `update_count` | `end_update - start_update + 1` |
| `closed_by` | `probe`, `epoch_end`, `terminal` 중 window를 닫은 이유 |
| `train_wall_time_ns` | 해당 update 구간의 host wall time. 이전 evaluation 시간은 포함하지 않음 |
| `train_device_time_ns` | 선택값. GPU profile mode의 device elapsed time; 비활성화면 빈값 |
| `eval_wall_time_ns` | window를 닫은 뒤 수행한 모든 evaluation의 host wall time; 없으면 빈값 |
| `eval_device_time_ns` | 선택값. 위 evaluation의 device elapsed time; 비활성화면 빈값 |

GPU 일반 기록 모드는 update마다 stream을 synchronize하지 않는다. `train_device_time_ns`와
`eval_device_time_ns`는 CUDA event 등 backend event를 지원하는 profile mode에서만
기록하며, window 종료에서 한 번 동기화해 확정한다. `*_wall_time_ns`는 host dispatch와
대기 시간을 포함한 wall time이다.

MLflow mapping:

| CSV 열 | MLflow metric | MLflow step |
| --- | --- | --- |
| `train_wall_time_ns` | `runtime/window/train_wall_time_ms` | `end_update` |
| `train_device_time_ns` | `runtime/window/train_device_time_ms` | `end_update` |
| `eval_wall_time_ns` | `runtime/window/eval_wall_time_ms` | `end_update` |
| `eval_device_time_ns` | `runtime/window/eval_device_time_ms` | `end_update` |

MLflow에는 ns 값을 ms로 변환해 기록한다. 빈 device/evaluation 값은 metric을 보내지
않는다.

## 관찰 그룹

### GO01 — optimizer trajectory

파일: `observations/trajectory.csv`

| 열 | 기록 시점 |
| --- | --- |
| `update` | 1–30 |
| `x`, `y` | 해당 update **전** optimizer state |
| `objective` | 같은 `(x,y)`에서 계산한 `x²/20+y²` |
| `grad_x`, `grad_y` | 같은 `(x,y)`에서 계산한 gradient |

MLflow에는 `update/trajectory/x`, `update/trajectory/y`, `update/trajectory/objective`를 update step으로 batch 기록한다.

### GO02 — activation histogram

파일: `observations/activation_histogram.csv`

| 열 | 기록 시점 |
| --- | --- |
| `layer` | 1–5 |
| `bin_index`, `bin_left`, `bin_right` | 고정 histogram bin 정의 |
| `count` | 해당 layer activation의 bin count |
| `sample_count` | histogram에 넣은 activation 원소 수 |

파일: `observations/activation_summary.csv`

| 열 | 기록 시점 |
| --- | --- |
| `layer` | 1–5 |
| `mean`, `std`, `min`, `max`, `zero_ratio` | 같은 forward pass에서 계산 |

MLflow에는 layer histogram 전체를 넣지 않는다. `observation/activation/layer_{n}/mean`, `observation/activation/layer_{n}/std`만 기록하고 histogram은 artifact로만 보존한다.

## 학습 그룹별 evaluation schedule

| 그룹 | `updates.csv` | `evaluations.csv` 기록 시점 |
| --- | --- | --- |
| `GT01` | update 1–2,000 모두 | 없음 |
| `GT02` | update 1–2,000 모두 | 없음 |
| `GT03` | 모든 update | update `1, 4, 7, …, 601` 뒤: `mnist-train-first-300`, `mnist-test-full` accuracy |
| `GT04` | 모든 update | 매 epoch 시작 직후 첫 update 뒤: `mnist-train-first-300`, `mnist-test-full` accuracy |
| `GT05` | 모든 update | update `1, 11, 21, …, 191` 뒤: `mnist-train-first-1000` accuracy |
| `GT06` | 모든 update | 매 epoch 시작 직후: `mnist-train-first-1000`, `mnist-test-first-1000` accuracy; 종료 후 `mnist-test-full` accuracy |
| `GT07` | 모든 update | 매 epoch 시작 직후: `mnist-train-first-1000`, `mnist-test-first-1000` accuracy; 종료 후 `mnist-test-full` accuracy |
| `GT08` | 모든 update | update `20, 40, …` 뒤: `mnist-train-first-1000`, `mnist-test-first-1000` accuracy; 매 epoch 종료 후: `mnist-test-full` accuracy |

`GT01`과 `GT02`의 loss graph는 `updates.csv`의 `update/loss`에서 직접 만든다. 원본형 smoothing이 필요하면 분석 단계에서 그 raw loss에 적용한다.

## 구현 경계

```text
Trainer
  매 update: UpdateEvent 발행
  명시 요청: evaluation 계산과 EvaluationResult 반환

Evaluator
  executor가 지정한 evaluation_set_id에서 loss/accuracy 계산

Experiment executor
  UpdateEvent → updates.csv record 생성
  지정 schedule: evaluator 호출과 evaluations.csv record 생성
  probe/epoch/terminal 경계: timing_windows.csv record 생성
  checkpoint: checkpoints.csv record 생성

Observation runner
  GO01 trajectory.csv 또는 GO02 activation artifacts 생성

MLflow sink
  위 CSV record를 256개 단위로 log_batch
```
