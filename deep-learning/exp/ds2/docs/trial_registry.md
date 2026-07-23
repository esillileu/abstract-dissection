# 2권 원본 시행(trial) 기록 레지스트리

대상 원본은 [deep-learning-from-scratch-2](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/)다. 이 문서는 고정된 데이터·모델·파라미터·학습 절차의 한 번 실행만 다룬다.

- 분석용 평균/표준편차, AUC, threshold 도달, paired difference, 최종 모델 순위는 포함하지 않는다.
- 구현·데모·단순 표시 예제는 제외했다. 표의 값은 원본이 실제 학습 중 또는 평가에서 기록·출력·그린 raw 값이다.
- source-default와 source에 명시된 모델·입력 방향 선택 조건, 관찰 시행을 전개하면 15개다. 실제 run ID는 `<trial_id>.seed-<n>`으로 만든다.

## 공통 raw record

| 원본의 기록 방식 | 보존할 raw record |
| --- | --- |
| interval 평균 loss/PPL | `update`, `epoch`, interval 시작/종료 update, `train/nll` 또는 `train/objective`, `train/perplexity`, `token_count` |
| epoch별 생성 정확도 | `update`, `epoch`, `split`, `evaluation_set_id`, `sequence_count`, `exact_match_accuracy` |
| checkpoint selection | `epoch`, valid PPL, best 여부, lr, checkpoint ID |
| 대표 생성 출력 | 고정 `example_id`, source, target, prediction, exact-match |

## 시행

| 시행 ID | 원본 실행·고정 조건 | 원본이 기록한 값과 시점 |
| --- | --- | --- |
| `dlfs2.ch03.toy-cbow-full-softmax` | [`ch03/train.py`](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch03/train.py); toy sentence, window `1`, vocab `7`, SimpleCBOW embedding `5`, Adam `.001`, batch `3`, `1,000` epochs | Trainer interval 평균 train loss; 최종 word vector checkpoint/artifact |
| `dlfs2.ch04.ptb-cbow-negative-sampling` | [`ch04/train.py`](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch04/train.py); PTB train, CBOW, window `5`, embedding `100`, negative sample `5`, Adam `.001`, batch `100`, `10` epochs | Trainer interval 평균 train loss와 elapsed time; 종료 때 word vectors와 word↔ID dictionary checkpoint |
| `dlfs2.ch04.ptb-skipgram-negative-sampling` | 같은 [`ch04/train.py`](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch04/train.py)의 명시된 `SkipGram` model 선택; PTB train, window `5`, embedding `100`, negative sample `5`, Adam `.001`, batch `100`, `10` epochs | CBOW와 같은 cadence의 Trainer interval 평균 train loss와 elapsed time; 별도 SkipGram word-vector checkpoint |
| `dlfs2.ch05.ptb-small-rnnlm` | [`ch05/train.py`](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch05/train.py); PTB train 첫 `1,000` tokens, SimpleRnnlm embedding/hidden `100`, SGD `.1`, batch `10`, BPTT `5`, `100` epochs | RnnlmTrainer interval 평균 NLL에서 계산한 train PPL을 list·콘솔에 기록 |
| `dlfs2.ch06.ptb-lstm-rnnlm` | [`ch06/train_rnnlm.py`](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch06/train_rnnlm.py); PTB full train/test, LSTM Rnnlm embedding/hidden `100`, SGD `20`, batch `20`, BPTT `35`, clip `.25`, `4` epochs | 20 update마다 train PPL; 종료 후 full test PPL 한 번; `Rnnlm.pkl` checkpoint |
| `dlfs2.ch07.addition.seq2seq-forward` | [`ch07/train_seq2seq.py`](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch07/train_seq2seq.py); `addition.txt` 90/10 split, reverse false, vanilla Seq2seq, embedding `16`, hidden `128`, Adam `.001`, batch `128`, clip `5`, `25` epochs | 매 epoch 전체 test set sequence exact-match accuracy; 처음 10 test example의 question/target/prediction |
| `dlfs2.ch07.addition.seq2seq-reverse` | 같은 source; reverse true, vanilla Seq2seq. 나머지 data/model size/optimizer/budget 동일 | 매 epoch 전체 test sequence exact-match accuracy; 처음 10 test example의 question/target/prediction |
| `dlfs2.ch07.addition.peeky-seq2seq-forward` | 같은 source; reverse false, PeekySeq2seq. 나머지 조건 동일 | 매 epoch 전체 test sequence exact-match accuracy; 처음 10 test example의 question/target/prediction |
| `dlfs2.ch07.addition.peeky-seq2seq-reverse` | 같은 source; reverse true, PeekySeq2seq. 나머지 조건 동일 | 매 epoch 전체 test sequence exact-match accuracy; 처음 10 test example의 question/target/prediction |
| `dlfs2.ch08.date.attention-seq2seq-reverse` | [`ch08/train.py`](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch08/train.py); `date.txt` 90/10 split, reverse true, AttentionSeq2seq, embedding `16`, hidden `256`, Adam `.001`, batch `128`, clip `5`, `10` epochs | 매 epoch 전체 test sequence exact-match accuracy와 처음 10개 생성 예제; 종료 checkpoint. attention map 재생성에 필요한 fixed example·decode policy도 checkpoint와 함께 보존 |
| `dlfs2.ch08.date.seq2seq-reverse` | 같은 [`ch08/train.py`](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch08/train.py)의 명시된 Vanilla Seq2seq model 선택; date split, reverse true, embedding `16`, hidden `256`, Adam `.001`, batch `128`, clip `5`, `10` epochs | 매 epoch 전체 test sequence exact-match accuracy와 처음 10개 생성 예제; 종료 checkpoint |
| `dlfs2.ch08.date.peeky-seq2seq-reverse` | 같은 source의 명시된 PeekySeq2seq model 선택; date split, reverse true, embedding `16`, hidden `256`, Adam `.001`, batch `128`, clip `5`, `10` epochs | 매 epoch 전체 test sequence exact-match accuracy와 처음 10개 생성 예제; 종료 checkpoint |
| `dlfs2.ch08.attention-alignment` | [`ch08/visualize_attention.py`](../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch08/visualize_attention.py); 학습된 AttentionSeq2seq checkpoint, test selection seed 1984, 5 examples | example별 decoder step × encoder position attention weight와 source/target 문자 label |

## 새 확장 시행

| 시행 ID | 고정 조건 | raw 기록 |
| --- | --- | --- |
| `ext.ds2.toy-skipgram-full-softmax` | toy sentence, window 1, vocab 7, SimpleSkipGram embedding 5, Adam .001, batch 3, 1,000 epochs | 20 update interval mean loss, word-vector checkpoint |
| `ext.ds2.ptb-word2vec-full-softmax` | PTB train, window 5, embedding 100, Adam .001, batch 100, 10 epochs; CBOW와 Skip-gram full softmax | 20 update interval mean loss, word-vector checkpoint |
| `ext.ds2.ptb-lm-recipe-comparison` | PTB train/valid/test, embedding/hidden 650, SGD 20, batch 20, BPTT 35, clip .25, max 40 epochs; Rnnlm/LSTM Rnnlm/BetterRnnlm | 20 update train PPL, epoch valid PPL, selected-checkpoint terminal test PPL |

## 시행 선택의 범위

`ch04/train.py`의 SkipGram, `ch07/train_seq2seq.py`의 Peeky·reverse, `ch08/train.py`의 Vanilla·Peeky는 source에서 한 줄을 전환해 선택하는 책의 실행 조건이므로 위에서 독립 시행으로 전개했다. 별도의 학습 entrypoint가 없는 `ch03/simple_skip_gram.py`은 원본 시행에는 포함하지 않고, GT01에서 CBOW와 같은 조건을 적용한 새 확장 시행으로 등록했다.
