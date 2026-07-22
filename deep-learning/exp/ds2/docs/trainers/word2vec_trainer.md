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
- forward → backward → optional clip → optimizer update 뒤, **동일 immutable input**으로 post-update mean objective를 재계산해 `UpdateEvent.loss`를 만든다.
- negative sampling의 post-update 재계산은 새 negative를 draw하지 않는다. update 때 쓴 candidate set을 재사용해 training RNG와 후속 batch를 바꾸지 않는다.
- `UpdateEvent.batch_size`는 실제 context row 수, `lr`은 해당 update에 적용한 값이다.
- `SourceObjectiveSample`에는 update 전 objective와 원본 local zero-based iteration을 넣는다. executor가 `iters % eval_interval == 0` 규칙 그대로 interval mean loss point를 만든다.

full softmax와 negative sampling은 objective 정의가 다르다. 공통 Trainer는 둘의 raw loss를 비교·정규화·순위화하지 않으며, `resolved_config.objective_id`만 보존한다.

## source curve와 평가

| 그룹 | source curve | evaluator |
| --- | --- | --- |
| `GT01` | 책 Trainer의 zero-based interval mean train loss | 없음 |
| `GT02` | 책 Trainer의 zero-based interval mean train loss | 없음 |

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
