# DS2 실험용 Trainer 공통 요구사항

이 문서는 2권의 `Word2VecTrainer`, `LanguageModelTrainer`, `Seq2seqTrainer`가 공유하는 계약이다. 태스크별 요구사항은 [Word2Vec](trainers/word2vec_trainer.md), [Language Model](trainers/language_model_trainer.md), [Seq2seq](trainers/seq2seq_trainer.md)에 둔다.

기준은 [실행 그룹](execution_groups.md), [시행 기록 스키마](recording_schema.md), [GT MLflow 스키마](mlflow/gt_common.md)다. 기존 Trainer 구현은 사용하지 않고 새로 구현한다.

## 책임 경계

`ExperimentExecutor`가 실행 명세의 소유자다. Trainer는 학습 사실과 명시적으로 요청된 평가 결과만 event로 반환하며, group ID·CSV·MLflow·checkpoint 정책을 알지 않는다.

| 책임 | 담당 |
| --- | --- |
| group별 config, seed, dataset/split, sampler, budget, schedule 결정 | `ExperimentExecutor` |
| task batch, forward/backward/update, counter, model state | 각 Trainer |
| source curve aggregate, evaluation trigger, timing window, CSV/MLflow | executor와 `RecordSink` |
| checkpoint 저장·hash·resume 검증 | executor 또는 `CheckpointManager` |
| attention map 추출 | GO01 observation runner |

## 공용 event/API

| 구성요소 | 최소 내용 |
| --- | --- |
| `UpdateEvent` | `update`, `epoch`, 실제 `batch_size`, post-update mean `loss`, 적용 `lr` |
| `SourceObjectiveSample` | update 전 objective, unit 수, source curve reducer에 필요한 local iteration/epoch 정보 |
| `EpochEvent` | 완료 epoch, update 범위, 실제 처리 수 |
| `TrainEndEvent` | 종료 이유(`completed`, `max_updates`, `stopped`, `error`)와 최종 counter |
| `EvaluationResult` | `unit`, `unit_count`, metric→value. metric은 `loss`, `accuracy`, `perplexity`, `exact_match_accuracy`, `token_accuracy` |
| `TrainerState` | global update, epoch, task가 소유한 재개 가능 state |
| `evaluate(source, metrics=...)` | executor가 준 고정 source에 대한 mode-safe 평가 |

- 성공 update마다 counter 증가 뒤 정확히 하나의 `UpdateEvent`를 발행한다.
- `UpdateEvent`만으로 `updates.csv` 행을 만들 수 있어야 한다.
- `SourceObjectiveSample`은 원본의 pre-update loss/PPL curve를 위한 내부 event다. `updates.csv`에는 저장하지 않고 executor가 `observations/source_curves.csv` point로 집계한다.
- GPU 일반 모드에서는 event마다 host scalar materialization 또는 synchronize를 하지 않는다. RecordSink가 device-side buffer를 최대 256개 모아 전송한다.

## 필수 공통 동작

1. `max_epochs`와 `max_updates`를 모두 지원하고, 먼저 도달한 budget에서 멈춘다.
2. forward → backward → 선택적 gradient clipping → optimizer update를 하나의 successful update로 정의한다.
3. DS2 공통 raw loss는 update 뒤 동일 batch를 다시 계산한 mean objective다. 재계산은 training RNG, sampler state, recurrent state, model mode를 바꾸지 않아야 한다.
4. evaluator는 train sampler/RNG를 소비하지 않으며, eval mode와 상태를 호출 전 값으로 복원한다.
5. trainer는 `state_dict()`/`load_state_dict()`로 자신이 소유한 counter·cursor·recurrent state·sampler state를 재개 가능하게 제공한다. executor 소유 state는 executor checkpoint에 둔다.
6. Trainer는 CSV, MLflow, manifest, timing 측정을 직접 쓰지 않는다.

## Executor/RecordSink 요구사항

- `UpdateEvent`를 `updates.csv`와 `update/train/loss`, `update/train/lr`로 변환한다.
- `SourceObjectiveSample`을 source curve schedule에 따라 집계해 `observations/source_curves.csv`로 쓴다. `plot_index`는 원본 list의 0-based append 순서다.
- executor만 evaluation schedule을 판정하고 `EvaluationResult`를 `evaluations.csv` long-form row로 만든다.
- source-curve/evaluation probe 사이의 train 시간과 그 뒤 evaluation 시간을 분리해 `timing_windows.csv`에 기록한다.
- record는 256 rows, epoch 끝, checkpoint 직전, 정상/예외 종료에 durable flush한다. MLflow 실패로 CSV를 되돌리거나 trainer update를 재실행하지 않는다.
- checkpoint 전에 sink를 flush하고 model·optimizer·trainer·executor/sampler state, config/dataset/split digest를 함께 검증한다.

## DS1 정책 적용과 의도적 차이

| DS1 정책 | DS2 결정 | 이유 |
| --- | --- | --- |
| executor가 schedule/CSV/MLflow/checkpoint 소유 | 그대로 적용 | 세 Trainer가 group 조건문 없이 공용 실행 lifecycle을 쓴다. |
| `updates.csv.loss`는 post-update mean | 그대로 적용 | DS1과 같은 공용 raw update history를 유지한다. |
| Trainer event를 공식 기록 경로로 사용 | 그대로 적용 | callback 내부 list/history 변환을 사용하지 않는다. |
| evaluation은 executor 주도, RNG 비소비 | 그대로 적용 | fixed probe와 재현성을 보장한다. |
| probe-to-probe timing, GPU 동기화 최소화 | 그대로 적용 | `timing_windows.csv`와 동일한 runtime 의미를 유지한다. |
| 지도 분류 `EvaluationResult(loss, accuracy)` | 확장 | LM은 token PPL, Seq2seq는 greedy sequence exact match가 필요하다. |
| update event만으로 책 그래프 생성 | 보완 | 책의 Word2Vec/LM graph는 pre-update objective를 사용하므로 `SourceObjectiveSample`과 source-curve artifact를 추가한다. |

기존 DS1/현 구현 Trainer는 재사용하지 않는다. 이 문서는 DS1의 **정책과 artifact 경계**만 재사용한다.
