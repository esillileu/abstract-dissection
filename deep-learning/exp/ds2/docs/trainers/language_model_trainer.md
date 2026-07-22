# LanguageModelTrainer 요구사항

대상은 `GT03`–`GT05`의 PTB RNNLM/LSTM/BetterRnnlm 학습이다. 공통 계약은 [DS2 Trainer 공통 요구사항](../trainer_requirements.md)을 따른다.

## 지원 범위

| 기능 | 요구사항 |
| --- | --- |
| model adapter | SimpleRnnlm, Rnnlm, LSTM Rnnlm, BetterRnnlm |
| batch | corpus의 offset/jump 기반 `(batch_size, time_size)` truncated-BPTT batch |
| state | 학습 중 recurrent state 유지, update 뒤 truncated history detach |
| objective | token mean NLL, source curve와 evaluator에서 PPL은 `exp(token-weighted mean NLL)` |
| clipping | config가 요구할 때 optimizer pre-step transform에서 global gradient norm clipping |

Trainer는 model class나 `GT03`–`GT05` ID로 schedule을 분기하지 않는다. batch offset과 BPTT size만 실행하고, clipping은 executor가 resolved config로 조립한 optimizer transform이 소유한다.

## update와 recurrent-state 안전성

- successful update 뒤 `UpdateEvent`에는 동일 BPTT batch의 post-update mean NLL과 적용 lr을 넣는다.
- post-update NLL 재계산은 forward 전 recurrent state를 snapshot하고, 재계산 뒤 학습이 계속될 recurrent state를 정확히 복원한다. 이 재계산은 time cursor, dropout RNG, sampler RNG를 소비하거나 바꾸지 않는다.
- update 전 token mean NLL과 실제 token 수는 `SourceObjectiveSample`로 발행한다. executor가 source curve의 PPL을 token-weighted 방식으로 계산한다.
- epoch/중단/재개 경계에서 corpus cursor와 recurrent state의 처리 순서를 config로 고정한다. validation/test evaluation 전후에는 model state를 reset하고, 호출 전 학습 state를 복원한다.

## evaluator

`evaluate(source, metrics={perplexity})`는 고정 sequential corpus를 사용한다.

- full corpus를 time order대로 순회하고 token 수로 가중한 mean NLL에서 PPL을 한 번 계산한다.
- chunk NLL은 device `float64`로 순차 누적하고, mean NLL과 PPL을 evaluation 종료에서 한 번에 host로 전송한다.
- evaluation은 train batch cursor, training recurrent state, training RNG를 바꾸지 않는다.
- 반환은 `EvaluationResult(unit=token, unit_count=..., metrics={perplexity: ...})`다.

| 그룹 | source curve | evaluator |
| --- | --- | --- |
| `GT03` standard | 원본 `iters % 20 == 0` interval train PPL | 없음 |
| `GT03` custom | 매 epoch train PPL | 없음 |
| `GT04` | 원본 zero-based interval train PPL | terminal full-test PPL |
| `GT05` | 원본 BetterRnnlm zero-based interval train PPL console series | 매 epoch valid PPL, selected checkpoint terminal test PPL |

`GT05`의 valid PPL 비교, lr `/ 4`, selected checkpoint 결정은 executor policy다. Trainer는 PPL만 반환한다.

## 재개와 검증

- `TrainerState`에는 global update, epoch, epoch 내 iteration, time cursor, recurrent state, batch offset 정의와 소유 RNG state를 포함한다.
- 중단·재개 후 BPTT batch 순서, update loss, source PPL point, valid checkpoint 선택이 중단 없는 실행과 동일해야 한다.
- source curve trigger는 local zero-based iteration을 기준으로 검증한다. 단순히 global update 20의 배수로 바꾸면 안 된다.

## 완료 조건

- PPL은 batch 평균의 단순 평균이 아니라 token-weighted NLL에서 계산한다.
- post-update record와 source pre-update PPL 계산이 recurrent state를 변경하지 않는다.
- evaluation 전후 training mode, cursor, recurrent state가 동일하다.
