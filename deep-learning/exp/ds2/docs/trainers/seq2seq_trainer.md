# Seq2seqTrainer 요구사항

대상은 `GT06`, `GT07`의 addition/date Seq2seq 학습이다. 공통 계약은 [DS2 Trainer 공통 요구사항](../trainer_requirements.md)을 따른다.

## 지원 범위

| 축 | 지원 값 |
| --- | --- |
| architecture | Vanilla Seq2seq, Peeky Seq2seq, Attention Seq2seq |
| input transform | forward 또는 reverse input |
| batch | resolved config의 epoch sampler와 batch size를 따르는 sequence batch |
| training | teacher-forced loss, optional gradient clipping |
| evaluation | greedy decode, full-test sequence exact match, optional token accuracy |

Trainer는 architecture/input reversal/group ID로 evaluation schedule을 분기하지 않는다. executor가 model adapter, transform, fixed prediction IDs를 전달한다.

## update 요구사항

- forward → backward → optional clip → optimizer update 뒤 동일 batch의 post-update mean objective를 다시 계산해 `UpdateEvent.loss`로 발행한다.
- 재계산은 training sampler/RNG, dropout RNG, model train/eval mode, encoder/decoder state를 바꾸지 않는다.
- `UpdateEvent.batch_size`는 실제 sequence example 수다. remainder batch 허용 여부는 resolved config에 고정한다.
- source curve용 loss는 요구하지 않는다. 책의 accuracy graph는 evaluator 결과로 만들며, executor가 그 값을 `source_curves.csv`에도 `plot_index=epoch-1`로 저장한다.

## greedy evaluator와 prediction

`evaluate(source, metrics={exact_match_accuracy, token_accuracy})`는 fixed sequential test source와 greedy decode를 사용한다.

- model eval mode에서 encoder/decoder state를 example마다 초기화한다.
- EOS/문자열 정규화, input reverse, decode maximum length는 resolved config에 고정한다.
- full test의 `correct_sequence_count / sequence_count`를 `exact_match_accuracy`로 반환한다.
- token accuracy를 기록하면 target token 수와 padding/EOS 포함 규칙을 config에 명시한다.
- evaluator는 training sampler/RNG와 model의 학습 state를 바꾸지 않는다.

| 그룹 | 평가/원본 graph | 추가 artifact |
| --- | --- | --- |
| `GT06` | 매 epoch full-test exact match, `plot_index=epoch-1` | fixed first 10 prediction rows |
| `GT07` | 매 epoch full-test exact match, `plot_index=epoch-1` | fixed first 10 prediction rows; Attention checkpoint |

GO01 attention heatmap은 학습 Trainer가 생성하지 않는다. observation runner가 matching-seed Attention checkpoint를 읽어 고정 5개 example의 weight와 rendering metadata를 기록한다.

## 재개와 검증

- `TrainerState`에는 global update, epoch, epoch 내 batch cursor, sampler RNG state와 model이 소유한 재개 state를 포함한다.
- evaluation/prediction이 batch-order RNG를 소비하지 않아야 한다.
- 재개 전후 epoch별 exact-match, fixed prediction rows, `plot_index`가 중복·누락 없이 일치해야 한다.

## 완료 조건

- Trainer가 성공 update마다 하나의 post-update `UpdateEvent`를 낸다.
- evaluator의 full-test exact match와 `source_curves.csv`의 같은 epoch 값이 정확히 같다.
- GO01 실행 여부가 GT07 학습 업데이트·RNG·checkpoint 내용을 바꾸지 않는다.
