# 실험 재구성 지침

이 지침은 1권에서 확정한 시행 재구성 방식을 2권에도 동일하게 적용하기 위한 기준이다. 목표는 원본 그래프를 raw history에서 재현하고, 이후 분석을 별도 단계로 유지하는 것이다.

## 1. 문서 구조

각 책 도메인은 `exp/ds{n}/docs/` 아래에 다음을 둔다.

```text
trial_registry.md        # 원본 시행과 원본 기록 시점
execution_groups.md      # 구현 담당자용 group/atomic protocol
recording_schema.md      # CSV raw record와 그룹별 평가 schedule
mlflow/
  README.md
  gt_common.md           # 학습 그룹 공용 MLflow/artifact schema
  go{nn}_*.md            # 관찰 그룹별 전용 schema
```

역할을 섞지 않는다.

| 문서 | 소유 내용 |
| --- | --- |
| `trial_registry.md` | 원본 시행 ID, 원본의 데이터·모델·파라미터·기록 시점 |
| `execution_groups.md` | 새 구현의 group ID, atomic ID, resolved protocol, 변동 축 |
| `recording_schema.md` | 실제 저장 열, MLflow 이름, evaluation cadence |
| `mlflow/*.md` | 구현자가 따를 MLflow params/metrics/artifact 계약 |

## 2. 그룹 ID

학습을 수행하는 그룹은 `GTnn`, 학습 없이 관찰만 수행하는 그룹은 `GOnn`으로 식별한다.

```text
GT01, GT02, ...   # training group
GO01, GO02, ...   # observation group
```

그룹은 하나의 완결된 protocol이다. 같은 그룹의 원자 시행은 다음을 공유한다.

- dataset, split/subset, preprocessing
- 학습 budget, sampler, update 단위
- 평가 집합과 평가 cadence
- checkpoint와 paired-seed 정책

원자 시행 간 차이는 그룹에 명시한 변동 축으로만 제한한다. 모델 구조가 변하는 비교도 그 구조가 명시된 factorial 변동 축이면 하나의 그룹으로 둔다.

## 3. 원본 시행을 그룹으로 옮기는 절차

2권의 각 원본 실행을 아래 순서로 처리한다.

1. `trial_registry.md`에 원본 시행을 한 줄로 적는다.
   - 데이터, 모델, optimizer, budget, 원본 기록값과 기록 시점
2. 학습인지 관찰인지 분류한다.
   - 학습이면 `GTnn`, forward/trajectory/시각화용 원자료 생성이면 `GOnn`
3. 같은 protocol을 공유하고 선언된 hyperparameter/architecture 축만 다른 시행을 하나의 그룹으로 묶는다.
4. 그룹 안의 모든 조건에 stable `atomic_run_id`를 부여한다.
5. 해당 그룹의 raw record와 cadence를 `recording_schema.md`에 확정한다.
6. MLflow metric/artifact 계약을 `mlflow/`에 작성한다.

원본의 주석 전환으로 선택하는 조건도 책에서 비교하는 조건이면 별도 atomic run으로 등록한다. 구현 파일만 있고 독립 실행 entrypoint가 없는 코드는 시행으로 등록하지 않는다.

## 4. 공통 기록 모델

분석용 metric을 trainer가 만들지 않는다. trainer/evaluator/observation runner는 raw record만 만든다.

```text
Trainer            : 학습 update와 checkpoint
Evaluator          : 지정된 schedule의 accuracy/PPL/exact-match 평가
Observation runner : trajectory, histogram, attention 등 비학습 관찰
Analysis           : smoothing, final/best, AUC, 평균, CI, 비교
```

모든 학습 trial artifact:

```text
manifest.json
updates.csv
evaluations.csv
checkpoints.csv
checkpoints/final.*
```

### `updates.csv`

매 optimizer update 뒤 한 행을 기록한다.

```text
update,epoch,batch_size,loss,lr
```

- `update`: 완료된 optimizer update, 1부터 시작
- `loss`: post-update, same-batch, mean objective
- `lr`: 그 update에 실제 적용한 값

### `evaluations.csv`

평가 시점마다 split별 한 행을 기록한다.

```text
axis,axis_step,update,epoch,evaluation_set_id,split,example_count,loss,accuracy
```

- `axis`: `update`, `epoch`, `terminal`
- `axis_step`: 원본 그래프의 x축 값
- `evaluation_set_id`: full set과 fixed probe를 반드시 구분

평가의 정확한 cadence는 그룹별로 명시한다. 예: “20 update마다 앞 1,000개 train/test accuracy”, “매 epoch full test PPL”, “종료 시 full test exact-match”.

## 5. MLflow 규칙

MLflow는 scalar history의 조회 인덱스이고 CSV artifact가 완전한 history다.

| raw record | MLflow metric | step |
| --- | --- | ---: |
| update loss | `update/train/loss` | update |
| update lr | `update/train/lr` | update |
| update 평가 accuracy | `update/eval_{split}/accuracy` | update |
| epoch 평가 accuracy | `epoch/eval_{split}/accuracy` | epoch |
| terminal 평가 accuracy | `terminal/eval_{split}/accuracy` | 마지막 update |

PPL, exact-match, token accuracy 등도 같은 규칙으로 metric suffix만 바꾼다. `final/*`, `best/*`, AUC, CI, 순위는 기록하지 않는다.

MLflow param/tag에는 최소 `group_id`, `atomic_run_id`, `master_seed`, `model_signature`, `dataset_id`, `split_id`, `resolved_config_sha256`, `evaluation_schedule_id`를 둔다. 전체 config와 checksum은 `manifest.json`에 둔다.

## 6. I/O 규칙

값은 지정된 update/epoch에 즉시 계산한다. 저장만 buffer로 지연한다.

```text
매 event: 메모리 record buffer에 append
256 records / epoch 종료 / checkpoint 직전 / run 종료:
  CSV flush
  MLflow log_batch
```

buffering은 x축 step, 기록 시점, metric 값을 바꾸지 않는다. 따라서 원본 그래프 재현에 영향을 주지 않는다.

## 7. 고정 probe 평가

학습 중 full dataset 평가가 비싸면 고정 probe를 사용한다.

- evaluator는 순차 loader를 사용하고 training sampler/RNG를 소비하지 않는다.
- evaluator는 eval mode로 전환하고 gradient를 계산하지 않으며, 종료 후 training mode를 복원한다.
- probe 예: `mnist-train-first-1000`, `mnist-test-first-1000`
- full test 평가는 그룹 schedule에 따라 epoch 또는 terminal에 별도 기록한다.

## 8. 2권 적용 시 추가 사항

2권은 1권의 `accuracy` 외에 다음 evaluator/observation schema를 추가한다.

| 유형 | raw record 핵심 |
| --- | --- |
| Word2Vec | update loss, token/context count, word-vector checkpoint, fixed query rank/score artifact |
| RNNLM | update/interval NLL, token count, PPL, valid/test evaluator 결과, checkpoint 선택 event |
| Seq2seq | epoch/update exact-match, token accuracy, fixed example source/target/prediction artifact |
| Attention | fixed example ID, decode step, encoder position, attention weight artifact |
| Gradient observation | time step, gradient norm |

각 추가 유형은 `mlflow/go{nn}_*.md` 또는 필요한 `mlflow/gt_*.md`로 별도 작성한다. 공통 training history 형식은 바꾸지 않는다.

## 9. 완료 확인

각 2권 그룹은 아래 질문에 문서만 보고 답할 수 있어야 한다.

1. 어떤 atomic run을 실행하는가?
2. 모델·데이터·변동 축은 무엇인가?
3. 매 update에 어떤 열을 남기는가?
4. 언제, 어느 fixed/full evaluation set에서 무엇을 평가하는가?
5. MLflow metric 이름과 step은 무엇인가?
6. 어떤 artifact가 원본 그래프/관찰물을 복원하는가?
