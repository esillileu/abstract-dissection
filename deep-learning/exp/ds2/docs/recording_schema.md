# 시행 기록 스키마

목표는 원본 그래프를 같은 x축 좌표와 raw 값으로 재현하고, 분석 단계가 추가 통계를 계산할 수 있게 하는 것이다. 이 문서는 2권 `GT01`–`GT07`, `GO01`이 남겨야 하는 데이터와 기록 시점을 고정한다.

DS1의 공용 artifact·event·timing 계약을 따른다. 기존 Trainer 구현은 사용하지 않으며, 새 `Word2VecTrainer`, `LanguageModelTrainer`, `Seq2seqTrainer`와 공용 executor/Recorder가 이 계약을 구현한다.

MLflow metric 이름과 artifact layout의 구현 계약은 [`mlflow/`](mlflow/)에 둔다. 분석 단계는 이 문서의 raw record만 읽는다. smoothing, final/best, AUC, 평균, CI, 순위, paired difference는 여기서 기록하지 않는다.

## 저장 규칙

- 값은 지정된 update/epoch에서 계산한다.
- `updates.csv`, `evaluations.csv`, `timing_windows.csv`, `observations/source_objectives.csv`, `observations/source_curves.csv`는 메모리 buffer에 쌓아 256 records마다, epoch 종료, checkpoint 직전, run 종료에 아직 쓰지 않은 row만 append한다.
- buffer flush는 계산 step, 원본 graph `plot_index`, 값에 영향을 주지 않는다.
- MLflow에는 같은 값을 batch로 전송한다. artifact CSV가 완전한 history의 기준이다.

## 모든 run의 파일

```text
artifact/
  manifest.json
  updates.csv
  evaluations.csv
  timing_windows.csv           # GT 그룹만 사용
  checkpoints.csv
  observations/
    source_objectives.csv      # GT01-GT05 canonical pre-update objective
    source_curves.csv          # 원본 graph/console series가 있는 GT만
    predictions.csv            # Seq2seq GT만
    attention.csv              # GO01만
    attention_render.json      # GO01만
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
| `loss_phase` | 항상 `post_update` (`updates.csv.loss`) |
| `loss_reduction` | 항상 `mean` |
| `evaluation_schedule` | 아래 그룹별 schedule 전체 |
| `source_curve_schedule` | 원본 graph/console point의 trigger·reducer·x축 |
| `timing_config` | timing window 경계, wall/device time 측정 여부와 단위 |

### `updates.csv`

모든 GT 학습 group은 매 optimizer update 뒤 한 행을 기록한다.

```text
update,epoch,batch_size,loss,lr
```

| 열 | 의미 |
| --- | --- |
| `update` | 완료된 optimizer update 수. 1부터 시작 |
| `epoch` | 해당 update가 속한 epoch. 1부터 시작 |
| `batch_size` | 실제 batch example 수 |
| `loss` | update **후** 동일 batch에서 다시 계산한 mean objective |
| `lr` | 해당 update에 적용한 learning rate |

`unit_count`는 모든 2권 batch가 config에서 결정된다(Word2Vec prediction term, LM token, Seq2seq example). 가변 길이 batch를 도입할 때만 `resolved_config`의 schema version을 올리고 update event에 명시적으로 추가한다.

MLflow mapping:

| CSV 열 | MLflow metric | MLflow step |
| --- | --- | --- |
| `loss` | `update/train/loss` | `update` |
| `lr` | `update/train/lr` | `update` |

### `evaluations.csv`

평가마다 split별·metric별 한 행을 기록한다. DS1의 고정 `loss`/`accuracy` 열 대신, 2권의 PPL·sequence metric을 같은 evaluator contract로 수용하기 위해 metric을 long-form으로 둔다.

```text
axis,axis_step,update,epoch,evaluation_set_id,split,unit,unit_count,metric,value
```

| 열 | 의미 |
| --- | --- |
| `axis` | `update`, `epoch`, `terminal` 중 하나 |
| `axis_step` | update/epoch이면 해당 번호, terminal이면 마지막 update |
| `update`, `epoch` | 평가 직전 완료된 update와 평가 시점 epoch |
| `evaluation_set_id` | 고정 evaluator input 식별자 |
| `split` | `train`, `valid`, `test` 중 하나 |
| `unit`, `unit_count` | `example`, `token`, `sequence`과 실제 평가 수 |
| `metric`, `value` | `loss`, `accuracy`, `perplexity`, `exact_match_accuracy`, `token_accuracy`와 값 |

MLflow metric은 `axis/eval_{split}/{metric}`이고 step은 `axis_step`이다. 예: `epoch/eval_test/exact_match_accuracy`, `terminal/eval_test/perplexity`.

### `checkpoints.csv`

```text
update,epoch,kind,path,sha256,checkpoint_id,selection_metric,selection_value
```

`kind`는 `final`, `periodic`, `selected` 중 하나다. `selection_metric/value`는 validation-selected checkpoint에만 채운다.

### `timing_windows.csv`

모든 GT 학습 group은 executor가 닫는 학습 timing window마다 한 행을 기록한다. 기본 window는 한 source-curve/evaluation probe 종료 뒤 다음 probe 직전까지의 연속한 training update다. probe가 없는 group은 epoch 종료 또는 terminal에서 window를 닫는다.

```text
start_update,end_update,update_count,closed_by,train_wall_time_ns,train_device_time_ns,eval_wall_time_ns,eval_device_time_ns
```

| 열 | 의미 |
| --- | --- |
| `start_update`, `end_update` | window에 포함한 첫/마지막 완료 update. 둘 다 포함 범위 |
| `update_count` | `end_update - start_update + 1` |
| `closed_by` | `probe`, `epoch_end`, `terminal` 중 window를 닫은 이유. source curve point도 `probe`다. |
| `train_wall_time_ns` | 해당 update 구간 host wall time. 이전 evaluation 시간은 포함하지 않음 |
| `train_device_time_ns` | 선택값. GPU profile mode의 device elapsed time |
| `eval_wall_time_ns`, `eval_device_time_ns` | window를 닫은 뒤 수행한 모든 evaluation 시간. 없으면 빈값 |

GPU 일반 기록 모드는 update마다 synchronize하지 않는다. device time은 backend event profile mode에서만 window 종료에 한 번 동기화해 확정한다. MLflow에는 ns 값을 ms로 바꿔 `runtime/window/{train,eval}_{wall,device}_time_ms`로 `end_update` step에 기록한다.

### `observations/source_objectives.csv`

```text
update,epoch,local_iteration,objective,unit_count
```

GT01-GT05가 매 update 발행한 pre-update objective의 canonical history다. 분석 단계는
이 파일에서 원본 interval 또는 epoch reducer와 `plot_index`를 재구성한다.
`updates.csv.loss`는 post-update 값이므로 이 파일을 대체할 수 없다.

### `observations/source_curves.csv`

원본이 실제 graph 또는 console series에 append한 한 point다.

```text
series_id,plot_index,update_start,update_end,epoch_start,epoch_end,unit,unit_count,metric,reducer,value
```

- `plot_index`는 원본 `loss_list`/`ppl_list`/`acc_list`의 0-based append 순서다. 원본 그림 x축은 이 값으로 그린다.
- `update_start`, `update_end`는 해당 point에 실제 포함된 update 범위다. `iters % 20 == 0`의 첫 update 한 건도 그대로 남긴다.
- `reducer`는 `mean`, `token_weighted_mean`, `exp_token_weighted_mean`, `identity` 중 하나다.
- `source_objective`는 update 전 계산되는 책의 loss/PPL 원재료다. 이는 `updates.csv`의 post-update loss와 다른 값일 수 있으며, source curve 계산을 위해서만 executor에 전달한다.

이 파일은 실행 중 확인과 기존 consumer 호환을 위한 projection이다. GT01-GT05 분석의
기준 데이터는 `source_objectives.csv`이며, 분석 단계가 reducer와 좌표를 다시 검증한다.

MLflow mapping:

| metric | MLflow metric | step |
| --- | --- | --- |
| `loss` | `series/train/loss` | `plot_index` |
| `perplexity` | `series/train/perplexity` | `plot_index` |
| `exact_match_accuracy` | `series/eval_test/exact_match_accuracy` | `plot_index` |

### `observations/predictions.csv`

```text
epoch,example_id,source,target,prediction,exact_match,token_correct,token_count
```

`example_id` 집합과 greedy decode policy는 resolved config에 고정한다.

### `observations/attention.csv`와 rendering metadata

```text
example_id,decode_step,encoder_position,weight
```

`observations/attention_render.json`에는 `source_labels`, `target_labels`, input reversal, encoder/decoder 축 방향, y-axis inversion, color range(`vmin`, `vmax`)를 기록한다.

## 관찰 그룹

### GO01 — attention alignment

학습하지 않는다. matching-seed `SEQD-ATTN-REV` checkpoint에서 seed `1984`로 고른 고정 test 5개에 대해 `attention.csv`, `attention_render.json`, `predictions.csv`를 기록한다. attention weight 전체는 MLflow scalar로 보내지 않는다.

## 학습 그룹별 schedule

| 그룹 | `updates.csv` | `source_curves.csv` | `evaluations.csv` |
| --- | --- | --- | --- |
| `GT01` | 모든 update | 원본 Trainer의 zero-based interval mean train loss | 없음 |
| `GT02` | 모든 update | 원본 Trainer의 zero-based interval mean train loss; objective는 manifest로 구분 | 없음 |
| `GT03` | 모든 update | standard: 원본 interval train PPL; custom: epoch train PPL | 없음 |
| `GT04` | 모든 update | 원본 zero-based interval train PPL | terminal full-test PPL |
| `GT05` | 모든 update | 원본 BetterRnnlm의 zero-based interval train PPL console series | epoch valid PPL; selected-checkpoint terminal test PPL |
| `GT06` | 모든 update | epoch full-test exact-match (`plot_index = epoch - 1`) | epoch full-test exact-match; fixed first 10 predictions |
| `GT07` | 모든 update | epoch full-test exact-match (`plot_index = epoch - 1`) | epoch full-test exact-match; fixed first 10 predictions; attention checkpoint |

## 구현 경계

```text
Word2VecTrainer / LanguageModelTrainer / Seq2seqTrainer
  매 update: post-update loss와 source-curve 원재료를 포함한 UpdateEvent 발행
  명시 요청: evaluator가 사용할 model state 반환

Evaluator
  executor가 지정한 evaluation_set_id에서 loss/PPL/accuracy를 계산

Experiment executor + Recorder
  UpdateEvent → updates.csv record 생성
  source-curve schedule → observations/source_curves.csv record 생성
  지정 schedule: evaluator 호출과 evaluations.csv record 생성
  probe/epoch/terminal 경계: timing_windows.csv record 생성
  checkpoint: checkpoints.csv record 생성

Observation runner
  GO01 attention artifact 생성

MLflow sink
  위 CSV record를 256개 단위로 log_batch
```
