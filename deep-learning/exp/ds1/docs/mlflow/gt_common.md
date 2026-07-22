# GT 공용 MLflow 스키마

대상: `GT01`–`GT08`.

## MLflow params/tags

모든 trial은 다음을 param 또는 tag로 기록한다.

| 이름 | 값 |
| --- | --- |
| `group_id` | `GTnn` |
| `atomic_run_id` | 원자 시행 ID |
| `master_seed` | child seed |
| `model_signature` | 구조 식별자 |
| `dataset_id`, `split_id` | 데이터·평가 집합 식별자 |
| `resolved_config_sha256` | 전체 resolved config hash |
| `evaluation_schedule_id` | 적용된 evaluation cadence 식별자 |
| `loss_phase` | `post_update` |
| `loss_reduction` | `mean` |

전체 resolved config와 seed·split checksum은 `manifest.json` artifact에 기록한다.

## 매 update 기록

Trainer는 optimizer update가 끝난 직후 동일 batch에서 post-update loss를 계산한다.

| MLflow metric | step | 값 | artifact 열 |
| --- | ---: | --- | --- |
| `update/train/loss` | `update` | post-update mean objective | `updates.csv.loss` |
| `update/train/lr` | `update` | 적용 learning rate | `updates.csv.lr` |

`updates.csv` 열:

```text
update,epoch,batch_size,loss,lr
```

## Timing window 기록

executor는 probe 간 학습 시간 또는 epoch/terminal에서 닫힌 학습 시간을
`timing_windows.csv`에 기록한다. 이 파일은 trainer event와 executor schedule에서
생성되며, trainer가 MLflow 또는 CSV를 직접 쓰지 않는다.

| MLflow metric | step | artifact 열 |
| --- | ---: | --- |
| `runtime/window/train_wall_time_ms` | `end_update` | `train_wall_time_ns / 1_000_000` |
| `runtime/window/train_device_time_ms` | `end_update` | `train_device_time_ns / 1_000_000` |
| `runtime/window/eval_wall_time_ms` | `end_update` | `eval_wall_time_ns / 1_000_000` |
| `runtime/window/eval_device_time_ms` | `end_update` | `eval_device_time_ns / 1_000_000` |

device-time metric은 backend event profile mode에서만 보낸다. 일반 GPU 실행은 매
update 동기화 없이 window 종료에서만 시간을 확정한다. 전체 열 정의와 `closed_by`
의 의미는 [시행 기록 스키마](../recording_schema.md#timing_windowscsv)를 따른다.

## 평가 기록

Trainer는 schedule이 요구하는 update/epoch에 evaluator를 호출한다. evaluator는 고정 sequential probe loader를 사용하며 training sampler/RNG를 소비하지 않는다.

| 평가 축 | MLflow metric | step |
| --- | --- | ---: |
| update 기반 train accuracy | `update/eval_train/accuracy` | update |
| update 기반 test accuracy | `update/eval_test/accuracy` | update |
| epoch 기반 train accuracy | `epoch/eval_train/accuracy` | epoch |
| epoch 기반 test accuracy | `epoch/eval_test/accuracy` | epoch |
| 종료 full-test accuracy | `terminal/eval_test/accuracy` | 마지막 update |

평가 loss를 실제 계산하는 경우에는 같은 이름에서 suffix만 `accuracy` 대신 `loss`로 바꾼다.

`evaluations.csv` 열:

```text
axis,axis_step,update,epoch,evaluation_set_id,split,example_count,loss,accuracy
```

`evaluation_set_id` 예: `mnist-train-first-1000`, `mnist-test-first-1000`, `mnist-test-full`.

## Checkpoint와 artifact

`checkpoints.csv` 열:

```text
update,epoch,kind,path,sha256
```

필수 artifact:

```text
manifest.json
  updates.csv
  evaluations.csv
  timing_windows.csv
  checkpoints.csv
checkpoints/final.npz
```

## I/O

- update/evaluation/timing record는 메모리 buffer에 append한다.
- 256 records마다, epoch 종료, checkpoint 직전, run 종료에 CSV flush와 MLflow `log_batch`를 수행한다.
- flush 시점은 metric value와 MLflow step을 변경하지 않는다.
