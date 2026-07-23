# Word2VecTrainer 요구사항

대상은 `GT01`, `GT02`의 toy/PTB Word2Vec 학습이다. 공통 계약은 [DS2 Trainer 공통 요구사항](../trainer_requirements.md)을 따른다.

## 지원 범위

| 축 | 지원 값 |
| --- | --- |
| architecture | CBOW, Skip-gram |
| objective | full vocabulary softmax, negative sampling |
| data | precomputed `(contexts, targets)` 또는 동일 의미의 corpus batch adapter |
| sampler | 원본과 같은 epoch별 permutation, partial batch drop 여부는 resolved config |

Trainer는 `GT01`/`GT02` 또는 objective 이름으로 분기하지 않는다. executor가 model/objective adapter와 batch policy를 조립한다.

## update 요구사항

- batch의 context/target과 negative-sampling candidate set을 한 update의 immutable input으로 취급한다.
- 학습 objective는 책처럼 candidate/context prediction term을 합하고 batch example만 평균한다. 벡터화된 prediction-term mean도 표준 비교 metric으로 함께 계산한다.
- forward → backward → optional clip → optimizer update 뒤, **동일 immutable input**으로 post-update prediction-term mean `UpdateEvent.loss`와 책 objective `UpdateEvent.book_loss`를 만든다.
- negative sampling의 post-update 재계산은 새 negative를 draw하지 않는다. update 때 쓴 candidate set을 재사용해 training RNG와 후속 batch를 바꾸지 않는다.
- `UpdateEvent.batch_size`는 실제 context row 수, `lr`은 해당 update에 적용한 값이다.
- `SourceObjectiveSample`에는 update 전 표준 objective, 책 objective와 원본 local zero-based iteration을 넣는다. executor가 `iters % eval_interval == 0` 규칙 그대로 두 interval point를 만든다.
- objective scalar와 Skip-gram objective 합은 device에 유지한다. host scalar materialization은 RecordSink의 bulk flush에서만 수행한다.

full softmax와 negative sampling은 objective 정의가 다르다. 공통 Trainer는 둘의 raw loss를 비교·정규화·순위화하지 않으며, `resolved_config.objective_id`만 보존한다.

## 실행 최적화 계약

- Skip-gram은 `(batch, context_width)` target을 `(batch * context_width)` prediction term으로 펼쳐 objective를 한 번 실행한다. context별 Python loop를 사용하지 않는다.
- negative sampling은 펼친 prediction term 전체의 candidate를 한 번에 draw한다. post-update objective는 그 candidate block을 그대로 재사용한다.
- 동일 candidate가 주어지면 벡터화 전 context별 구현과 같은 book loss 및 embedding gradient를 유지한다. 표준 loss는 모든 prediction term의 mean projection이다. 최적화는 update 수와 기록 시점을 바꾸지 않으며, post-update 재계산은 RNG를 추가 소비하지 않는다.
- PTB context/target 전처리는 sliding window를 벡터화해 원본의 왼쪽 context 뒤 오른쪽 context가 오는 순서를 유지한다.
- recorder flush는 이미 기록한 행을 다시 쓰지 않고 새 canonical row만 append한다. CSV schema와 MLflow 매-update mapping은 바꾸지 않는다.

## source curve와 평가

| 그룹 | source curve | evaluator |
| --- | --- | --- |
| `GT01` | `series/train/book_loss`: 책 Trainer의 zero-based interval mean; `series/train/loss`: 항당 mean 진단값 | 없음 |
| `GT02` | `series/train/book_loss`: 책 Trainer의 zero-based interval mean; `series/train/loss`: 항당 mean 진단값 | 없음 |

- `plot_index`는 각 원본 loss list의 append 순서다. global update로 대체하지 않는다.
- Word vector 품질 평가는 Trainer의 책임이 아니다. terminal vector/dictionary checkpoint 저장은 executor가 한다.

## 재개와 검증

- `TrainerState`에는 global update, epoch, epoch 내 batch cursor, batch-order RNG state를 포함한다.
- negative sampler가 Trainer 소유면 sampler distribution/RNG state도 포함한다. executor 소유면 executor state에 둔다.
- 재개 전후 source curve point의 update 범위와 `plot_index`가 중복 또는 누락되지 않아야 한다.

## 완료 조건

- successful update 수와 `UpdateEvent` 수가 일치한다.
- negative sampling post-update loss 기록이 training RNG를 추가 소비하지 않는다.
- source curve의 첫 point와 이후 zero-based interval trigger가 원본과 일치한다.
