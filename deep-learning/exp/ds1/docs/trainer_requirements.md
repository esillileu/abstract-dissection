# DS1 실험용 Supervised Trainer 요구사항

이 문서는 `GT01`–`GT08`을 실행하는 experiment executor가 학습 중의 사실을
일관되게 기록할 수 있도록 `ForwardTrainer`가 제공해야 할 기능을 정리한다. 기준은
[실행 그룹](execution_groups.md), [시행 기록 스키마](recording_schema.md),
[GT MLflow 스키마](mlflow/gt_common.md)다.

대상은 외부 criterion을 사용하는 지도 분류 학습 trainer다. `GO01`의 목적함수
궤적과 `GO02`의 activation 관찰은 optimizer/observation runner의 책임이므로 이
trainer의 범위가 아니다. 다만 같은 record sink와 artifact lifecycle은 공유할 수
있어야 한다.

## 책임 경계

`ExperimentExecutor`가 실행 명세의 소유자다. group별 budget, sampler, dataset/split,
evaluation schedule, artifact/MLflow flush, checkpoint 정책 및 재개 절차를 해석하고
실행한다. trainer는 이를 결정하거나 artifact를 저장하지 않는다.

trainer는 batch loop에서 일어난 학습 사실을 구조화된 event로 발행하고, executor가
요청한 명시적 동작만 수행한다. 따라서 같은 `ForwardTrainer`를 서로 다른 DS1 실행
명세에 사용할 수 있으며, 새로운 그룹을 추가해도 trainer에 group ID 조건문을 넣지
않는다.

| 책임 | 담당 |
| --- | --- |
| group별 protocol·seed·dataset/split·sampler·budget 결정 | `ExperimentExecutor` |
| forward/backward/update, counter, 실제 batch/loss/lr 산출 | `ForwardTrainer` |
| 언제/어떤 set을 평가할지 결정 및 evaluation record 저장 | `ExperimentExecutor` |
| `updates.csv`/`evaluations.csv`/MLflow/manifest flush | `ExperimentExecutor` 또는 그 `RecordSink` |
| checkpoint 시점·저장·검증·resume 정책 | `ExperimentExecutor` 또는 `CheckpointManager` |

## 설계 목표

- trainer의 update event만으로 executor가 `updates.csv`의 모든 열을 만들 수 있다.
- trainer는 group별 schedule·CSV·MLflow·checkpoint 정책을 알지 않는다.
- 평가·기록 callback이 training sampler 또는 그 RNG stream을 소비하거나 학습 결과를
  바꾸지 않는다.
- 중단 뒤 executor가 재개할 때 trainer state가 필요한 counter를 제공한다.

## 공개 인터페이스

### 입력 모델

executor가 `fit`에 train source와 학습 제어값을 전달한다. 현재의 tensor 기반
`fit(x_train, t_train, ...)`는 유지할 수 있고, loader 기반 API를 추가한다면 그
adapter도 executor가 조립한다. trainer의 공개 event에는 최소한 다음 정보를 담는다.

| 구성요소 | 책임 |
| --- | --- |
| `UpdateEvent` | `update`, `epoch`, 실제 `batch_size`, post-update `loss`, 적용 `lr` |
| `EpochEvent` | 완료 epoch, 해당 epoch의 update 범위·sample 수, train 집계 metric |
| `TrainEndEvent` | 종료 이유(completed/max_updates/stopped/error), 최종 counter |
| `TrainerState` | `global_step`, epoch 및 trainer가 소유한 재개 가능 상태 |
| `EvaluationResult` | `example_count`, 선택적으로 계산한 mean `loss`와 `accuracy` |
| `TrainingWindowEvent` | update 범위와 training/evaluation wall/device time |
| `evaluate(...)` | executor가 지정한 고정 evaluation set에서 loss/accuracy를 계산하는 명시적 요청 API |

event는 dataclass 또는 확장 가능한 callback protocol으로 정의한다. callback은
부가 관찰용이 아니라 executor가 record를 받는 공식 경로가 될 수 있으므로, update
성공 직후와 counter 증가 뒤에 정확히 한 번 발행해야 한다.

`UpdateEvent.loss`는 sink가 요구할 때만 host scalar로 materialize한다. GPU 일반
모드에서 trainer가 event를 만들기 위해 매 update마다 device scalar를 Python `float`로
변환하거나 synchronize해서는 안 된다. executor의 device-side metric buffer가 최대
256개를 모아 한 번에 host로 전송한다.

## 필수 기능

### 1. 학습과 update 경계

- `max_epochs`와 `max_updates`를 모두 지원하고, 먼저 도달한 budget에서 멈춘다.
- `permutation_per_epoch` 및 `with_replacement` sampler를 지원한다. replacement
  sampler의 epoch당 update 수, 마지막 batch 처리, `drop_last`는 executor가 전달한다.
- forward → criterion → backward → (선택) gradient clipping → optimizer update를
  한 update로 정의하고, 성공적으로 update한 뒤에만 `global_update`를 증가한다.
- DS1 executor가 요청하면 매 update 뒤 동일 batch로 mean objective를 재계산하고
  `UpdateEvent`로 발행한다. DS1은 `loss_phase=post_update`, `loss_reduction=mean`만
  허용하며, executor가 이 event를 `updates.csv`로 변환한다.
- event에는 실제 적용 learning rate와 실제 batch example 수를 포함한다. optimizer가
  parameter group 또는 schedule을 지원하면 lr 표현(`single`, `per-group`)을 API에서
  명확히 정의한다.
- model train/eval mode 전환은 예외가 발생해도 원래 상태로 복원한다.

### 2. Executor 주도 평가

trainer는 schedule을 소유하지 않는다. executor는 `UpdateEvent`/`EpochEvent`를 받아
아래 trigger를 판정하고, 필요할 때 `trainer.evaluate(evaluation_set)`을 호출한다.
trainer가 제공하는 것은 mode-safe하고 sampler/RNG를 소비하지 않는 평가 계산뿐이다.

| 실행 그룹 | 필요한 trigger |
| --- | --- |
| GT01, GT02 | 없음 |
| GT03 | update 1, 4, 7, …, 601 뒤 train-first-300 및 test-full |
| GT04 | 각 epoch의 첫 update 뒤 train-first-300 및 test-full |
| GT05 | update 1, 11, 21, …, 191 뒤 train-first-1000 |
| GT06, GT07 | 각 epoch 첫 update 뒤 train/test first-1000, 종료 뒤 test-full |
| GT08 | update 20, 40, … 뒤 train/test first-1000; 각 epoch 종료 뒤 test-full |

- evaluator는 executor가 준 sequential evaluation source만 사용하며 training sampler
  state/RNG를 바꾸지 않는다.
- evaluation set별로 mean loss와 accuracy를 독립적으로 선택해 계산한다. accuracy의
  label 형식(class index/one-hot)과 binary/multiclass prediction 규칙도 공통 utility로
  고정한다.
- trainer는 `example_count`, `loss`, `accuracy`만 반환한다. executor가 `axis`,
  `axis_step`, `update`, `epoch`, `evaluation_set_id`, `split`을 결합해
  `EvaluationRecord`를 만든다. 측정하지 않은 metric은 빈값(`None`)으로 기록한다.
- 현재의 `record_first_*`, `record_epoch_evaluation`, interval boolean 조합과
  `graph_evaluations`/`validation_evaluations`은 trainer에서 제거하고 executor의
  schedule adapter로 옮긴다.

### 3. Record sink와 telemetry 경계

- trainer는 `UpdateEvent`를 발행할 뿐 `updates.csv`, `evaluations.csv`, MLflow 또는
  manifest를 직접 알거나 쓰지 않는다.
- executor의 sink가 event를 `update, epoch, batch_size, loss, lr` row와 canonical
  MLflow metric으로 변환한다. evaluation/checkpoint record도 executor가 만든다.
- sink는 update/evaluation record를 최대 256개까지 buffer에 둔 뒤 CSV append와
  MLflow `log_batch`를 수행한다. epoch 종료, checkpoint 직전, 정상 종료, 예외 종료
  시에도 flush한다.
- CSV가 원본 history다. MLflow 실패는 재시도 가능한 전송 오류로 분리하며, 이미
  durable하게 저장한 CSV record를 삭제하거나 trainer update를 재실행하지 않는다.

### 4. Checkpoint와 재개 지원

- checkpoint 시점·파일 저장·sha256 record·config/dataset 검증은 executor의 책임이다.
- trainer는 executor가 checkpoint에 넣을 `TrainerState`를 `state_dict()`로 제공하고
  `load_state_dict()`로 복원한다.
- trainer가 sampler를 직접 소유하는 API라면 epoch 내 batch 위치와 sampler RNG state도
  state에 넣는다. sampler가 executor 소유라면 그 state는 executor checkpoint에 둔다.
- executor가 checkpoint 저장 전 sink를 flush하도록 lifecycle event를 제공한다.

### 5. 재현성·관측성

- model initialization, batch order, evaluation probe, input transform의 seed stream
  결정과 manifest 기록은 executor가 맡는다.
- optimizer pre-step transform이 제공하는 gradient clipping, backend CPU/GPU 지원,
  `ProfilingConfig`, runtime/epoch throughput 및 memory profiling은 유지한다.
- profiler의 측정은 학습 record와 분리한다. profiling을 켜거나 끈 것이 sampler,
  evaluation schedule, loss 값에 영향을 주면 안 된다.

### 6. 학습 시간 window

실험은 update별 시간 또는 최소한 한 probe 종료 뒤 다음 probe 직전까지 수행한 학습의
시간을 기록할 수 있어야 한다. timing window의 경계와 저장 여부는 executor의 실행
명세가 결정한다.

- 기본 단위는 `start_update`, `end_update`, `update_count`, `train_wall_time_ns`를
  갖는 `TrainingWindowEvent`다. executor는 예를 들어 20-update probe 직전에
  window를 닫고 평가 시간과 분리해 기록한다.
- GPU에서 매 update마다 host scalar를 읽거나 stream을 synchronize해 시간을 재지
  않는다. 그 방식은 비동기 kernel execution을 직렬화한다.
- 정확한 GPU 경과 시간이 필요한 profile mode에서는 backend event(CUDA event 등)를
  window의 시작/끝에 기록하고, probe 직전 또는 window flush 시 한 번만 synchronize해
  `train_device_time_ns`를 확정한다. 일반 모드의 `train_wall_time_ns`는 host dispatch와
  대기 시간을 포함한 실행 wall time임을 metadata에 명시한다.
- 평가 자체의 시간은 `eval_wall_time_ns`/`eval_device_time_ns`로 별도 기록한다.
  따라서 “20 update 학습이 느린가”와 “그 뒤 full-test 평가가 느린가”를 구분할 수
  있다.
- timing event는 update/evaluation record와 같은 durable sink로 전달하되, timing을
  켜지 않은 실행에서 학습 결과나 evaluation cadence가 바뀌면 안 된다.

## 현재 `ForwardTrainer` 대비

| 영역 | 현재 제공 | DS1에 필요한 보완 |
| --- | --- | --- |
| update | epoch/update budget, 두 sampler, clipping, pre/post-update step loss | 정확히 한 번 발행하는 `UpdateEvent`, 실제 batch size/lr |
| 평가 | trainer 내부 boolean 조합과 tensor probe 평가 | executor가 호출하는 mode-safe `evaluate(...)`; trainer 내부 schedule 제거 |
| 기록 | 메모리 `logs`, `step_losses`, callback | executor sink가 쓸 구조화 event; trainer의 CSV/MLflow 책임 없음 |
| checkpoint | epoch 종료 callback과 제한적인 trainer state | executor가 저장할 sampler 경계별 `TrainerState` |
| 재현성 | 외부에서 batch-order seed 설정 | trainer가 소유한 counter/state의 명확한 직렬화 계약 |
| 시간 | profiler의 선택적 epoch/step 측정 | executor가 정한 probe 경계의 `TrainingWindowEvent`, GPU event 기반 선택 측정 |
| 호환성 | 기존 executor가 내부 list를 history로 변환 | event → DS1 row 변환 adapter를 executor에 둠 |

## 구현 계획

### 1단계 — trainer event 계약 추가

대상: `src/mlprosection/trainer/` 및 `src/mlprosection/events.py`.

1. 불변 dataclass `UpdateEvent`, `EpochEvent`, `TrainEndEvent`, `EvaluationResult`,
   `TrainingWindowEvent`를 정의한다.
2. 기존 `TrainerCallback`을 event receiver로 확장한다. `on_update`는 update 성공,
   counter 증가, post-update loss 계산이 끝난 뒤 정확히 한 번 호출한다.
3. `ForwardTrainer`의 `record_step_loss`, `logs`, `step_losses`는 새 event를 읽어
   기존 list를 채우는 호환 adapter로 한 release 동안만 유지한다.
4. update event에 실제 batch size와 update에 적용한 lr를 넣는다. DS1 executor는
   post-update loss mode를 요청하며, 다른 loss phase는 DS1 config validation에서
   거부한다.

검증:

- 성공 update 수와 `UpdateEvent` 수가 같고, `max_updates` 경계에서도 마지막 event가
  한 번만 발생한다.
- remainder batch의 실제 크기, epoch 번호, learning rate가 event에 정확히 들어간다.
- CPU 기존 trainer 테스트가 유지된다.

### 2단계 — trainer에서 실행 정책 제거

대상: `ForwardTrainer`와 `SupervisedClassificationExecutor`.

1. `record_first_*`, `record_epoch_evaluation`,
   `record_step_*_interval`, `graph_evaluations`, `validation_evaluations`,
   `on_epoch_checkpoint`를 deprecated adapter를 거쳐 제거한다.
2. `evaluate(evaluation_source, metrics=...) -> EvaluationResult`를 제공한다. 이 API는
   model eval mode를 복원하고 training sampler/RNG를 소비하지 않는다.
3. executor는 `UpdateEvent`/`EpochEvent`를 받아 DS1 schedule을 판정하고, 요구된
   evaluation source만 trainer에 전달한다.

검증:

- GT03–GT08의 trigger가 문서에 정한 update/epoch 좌표에서만 실행된다.
- 두 evaluation set이 하나의 trigger에서 평가될 때 동일한 `(update, epoch)`를
  기록한다.
- 평가 전후 model의 training mode와 training sampler RNG state가 동일하다.

### 3단계 — DS1 record sink와 GPU-safe metric buffer

대상: `src/mlprosection/experiment/`의 DS1 adapter와 MLflow sink.

1. `UpdateEvent`를 `updates.csv` row와 `update/train/loss`, `update/train/lr` metric으로
   변환한다. evaluator 결과는 executor가 `EvaluationRecord`로 조립한다.
2. `updates.csv`, `evaluations.csv`, `checkpoints.csv` writer와 MLflow batch sender를
   같은 canonical record로 구동한다. 256 records, epoch 종료, checkpoint 직전, 정상·예외
   종료에서 durable flush한다.
3. GPU loss scalar는 device-side buffer에 유지하고 flush에서 한 번에 host로 전송한다.
   queue가 찼을 때도 raw record는 drop하지 않으며, MLflow 전송 실패는 CSV 기록 뒤
   재시도 가능한 오류로 처리한다.

검증:

- 2,000 update 실행에서 `updates.csv` 행 수, update 번호, MLflow step이 모두 일치한다.
- 강제로 MLflow 오류를 내도 CSV row는 누락되지 않는다.
- CuPy integration test에서 일반 기록 모드가 update마다 synchronize하지 않는지 spy로
  확인한다.

### 4단계 — probe timing과 profiling

대상: trainer backend timing helper와 executor schedule adapter.

1. executor가 probe 직전 training window를 닫고, probe 완료 뒤 다음 window를 연다.
   `TrainingWindowEvent`에는 update 범위와 train/eval 시간을 분리해 넣는다.
2. 일반 모드는 `perf_counter_ns()` 기반 wall time만 기록한다. GPU profile mode에서만
   CUDA event 쌍을 이용해 device time을 측정하고, probe/window flush 시 한 번
   synchronize한다.
3. update별 정확한 device time은 opt-in 상세 profiling으로 한정한다. 기본 DS1 실행은
   probe-to-probe timing을 사용한다.

검증:

- 20-update probe schedule에서 window가 `1–20`, `21–40`처럼 빈틈·중복 없이 닫힌다.
- train과 eval 시간이 별도 row/metric으로 기록된다.
- timing off/on이 loss, update 순서, evaluation 좌표를 바꾸지 않는다.

### 5단계 — checkpoint/restart와 cutover

대상: executor checkpoint manager, schema artifact writer, 기존 실행기 adapter.

1. checkpoint 전 record sink를 flush하고, model·optimizer·trainer state와 executor/
   sampler state를 함께 저장한다.
2. 재개 시 config digest, dataset/split checksum, 마지막 durable update를 검증한다.
   checkpoint와 CSV가 불일치하면 append 대신 명시적으로 실패한다.
3. 저장 성공 뒤 checksum을 계산해 `checkpoints.csv`와 manifest에 기록한다.
4. generic `history.csv` 변환은 호환 산출물로만 남기고 DS1 분석은 새 raw CSV만 읽도록
   전환한다.

검증:

- 임의 update에서 중단·재개한 결과의 model/optimizer state와 raw history가 중단 없는
  실행과 동일하다.
- final/periodic checkpoint record에 path와 sha256이 존재한다.
- 기존 `ForwardTrainer` tensor 호출과 profiler 회귀 테스트가 통과한다.

## 완료 조건

- trainer는 매 successful update마다 한 번의 post-update `UpdateEvent`를 발행한다.
- executor는 event만으로 모든 GT의 `updates.csv`와 정확한 schedule의
  `evaluations.csv`를 만든다.
- executor는 event와 실행 명세만으로 manifest, checkpoint record, MLflow metric의
  이름과 step을 결정하며, trainer는 그 저장 형식을 알지 않는다.
- checkpoint와 MLflow 전송은 trainer를 변경하지 않고 executor/sink에서 동작한다.
- 20-update probe 같은 지정 경계마다 training 시간과 evaluation 시간을 분리해
  기록할 수 있고, GPU 일반 모드에서 update마다 강제 동기화하지 않는다.
- 기존 profiler 및 현재 tensor 기반 호출을 사용하는 테스트가 호환 adapter를 통해
  계속 통과한다.
