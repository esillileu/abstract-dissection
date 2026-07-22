# 실행 그룹 구현 명세

이 문서는 구현 담당자용 실행 명세다. 원본 출처·원본 시행 ID·책의 관측 방식은 여기서 다루지 않는다. 학습 그룹은 `GTnn`, 관찰 그룹은 `GOnn`으로 식별한다. 기록 파일·열·MLflow metric 이름과 cadence는 [`recording_schema.md`](recording_schema.md)에, 새 Trainer의 공통/태스크별 책임은 [`trainer_requirements.md`](trainer_requirements.md)에 고정한다.

## 그룹화 계약

같은 그룹의 원자 시행은 dataset/split/subset, 학습 budget, sampler, update 단위, 평가 cadence, checkpoint와 paired-seed 정책을 공유한다. 원자 시행 간 차이는 해당 그룹의 변동 축으로만 제한한다.

## 관찰 그룹

| 그룹 | protocol | 공통 조건 | 변동 축 | 원자 시행 수 |
| --- | --- | --- | --- | ---: |
| `GO01` | attention alignment | fixed trained attention checkpoint, date test examples, greedy decode | fixed example ID | 5 |

## 학습 그룹

| 그룹 | protocol | 공통 조건 | 변동 축 | 원자 시행 수 |
| --- | --- | --- | --- | ---: |
| `GT01` | toy Word2Vec | toy corpus, CBOW full softmax, window 1, embedding 5, Adam .001, batch 3, 1,000 epochs | 없음 | 1 |
| `GT02` | PTB Word2Vec | PTB train, window 5, embedding 100, Adam .001, batch 100, 10 epochs | architecture × objective | 4 |
| `GT03` | small-corpus RNNLM | PTB first 1,000 tokens, SimpleRnnlm 100/100, SGD .1, batch 10, BPTT 5, 100 epochs | training-loop implementation | 2 |
| `GT04` | PTB LSTM RNNLM | PTB train/test, LSTM 100/100, SGD 20, batch 20, BPTT 35, clip .25, 4 epochs | 없음 | 1 |
| `GT05` | PTB LM recipe comparison | PTB train/valid/test, embedding/hidden 650, SGD 20, batch 20, BPTT 35, clip .25, max 40 epochs | architecture | 3 |
| `GT06` | addition Seq2seq | addition split, embedding 16, hidden 128, Adam .001, batch 128, clip 5, 25 epochs | architecture × reverse input | 4 |
| `GT07` | date Seq2seq | date split, reverse input, embedding 16, hidden 256, Adam .001, batch 128, clip 5, 10 epochs | architecture | 3 |

## 원자 시행 ID

| 그룹 | atomic run ID |
| --- | --- |
| `GT01` | `W2V-TOY-CBOW-FULL` |
| `GT02` | `W2V-PTB-CBOW-NS`, `W2V-PTB-SKIPGRAM-NS`, `W2V-PTB-CBOW-FULL`, `W2V-PTB-SKIPGRAM-FULL` |
| `GT03` | `LM-SMALL-RNN`, `LM-SMALL-RNN-CUSTOM` |
| `GT04` | `LM-LSTM` |
| `GT05` | `LM-RNN-RECIPE`, `LM-LSTM-RECIPE`, `LM-BETTER-RECIPE` |
| `GT06` | `SEQA-VAN-FWD`, `SEQA-VAN-REV`, `SEQA-PEEKY-FWD`, `SEQA-PEEKY-REV` |
| `GT07` | `SEQD-VAN-REV`, `SEQD-PEEKY-REV`, `SEQD-ATTN-REV` |
| `GO01` | `ATTENTION-ALIGNMENT` |

## Atomic 조건

### GT01 — toy CBOW

| atomic run ID | architecture | objective |
| --- | --- | --- |
| `W2V-TOY-CBOW-FULL` | SimpleCBOW, window 1, embedding 5 | full vocabulary softmax |

### GT02 — PTB Word2Vec

| atomic run ID | architecture | objective |
| --- | --- | --- |
| `W2V-PTB-CBOW-NS` | CBOW, window 5, embedding 100 | negative sampling, 5 negatives |
| `W2V-PTB-SKIPGRAM-NS` | Skip-gram, window 5, embedding 100 | negative sampling, 5 negatives |
| `W2V-PTB-CBOW-FULL` | CBOW, window 5, embedding 100 | full vocabulary softmax |
| `W2V-PTB-SKIPGRAM-FULL` | Skip-gram, window 5, embedding 100 | full vocabulary softmax |

negative sampling과 full softmax는 objective 정의가 다르므로 raw loss를 직접 비교하지 않는다. 동일 architecture 안에서 objective별 runtime과 고정 query artifact를 비교한다.

### GT03 — small-corpus RNNLM

| atomic run ID | model | loop |
| --- | --- | --- |
| `LM-SMALL-RNN` | SimpleRnnlm, embedding/hidden 100 | RnnlmTrainer interval loop |
| `LM-SMALL-RNN-CUSTOM` | SimpleRnnlm, embedding/hidden 100 | custom epoch loop |

### GT04 — LSTM RNNLM

| atomic run ID | model | checkpoint policy |
| --- | --- | --- |
| `LM-LSTM` | one-layer LSTM Rnnlm, embedding/hidden 100 | terminal checkpoint |

### GT05 — PTB LM recipe comparison

| atomic run ID | model | checkpoint policy |
| --- | --- | --- |
| `LM-RNN-RECIPE` | Rnnlm, embedding/hidden 650 | best valid PPL; non-improvement → lr / 4 |
| `LM-LSTM-RECIPE` | LSTM Rnnlm, embedding/hidden 650 | best valid PPL; non-improvement → lr / 4 |
| `LM-BETTER-RECIPE` | BetterRnnlm, two-layer LSTM 650/650, dropout .5 | best valid PPL; non-improvement → lr / 4 |

### GT06 — addition Seq2seq

| atomic run ID | architecture | reverse input |
| --- | --- | --- |
| `SEQA-VAN-FWD` | Vanilla Seq2seq | false |
| `SEQA-VAN-REV` | Vanilla Seq2seq | true |
| `SEQA-PEEKY-FWD` | Peeky Seq2seq | false |
| `SEQA-PEEKY-REV` | Peeky Seq2seq | true |

### GT07 — date Seq2seq

| atomic run ID | architecture | reverse input |
| --- | --- | --- |
| `SEQD-VAN-REV` | Vanilla Seq2seq | true |
| `SEQD-PEEKY-REV` | Peeky Seq2seq | true |
| `SEQD-ATTN-REV` | Attention Seq2seq | true |

### GO01 — attention alignment

| atomic run ID | source checkpoint | example selection |
| --- | --- | --- |
| `ATTENTION-ALIGNMENT` | matching-seed `SEQD-ATTN-REV` checkpoint | 5 test examples, selection seed 1984 |

## 원본 시행 ID

| 구현 그룹 ID | 원본 시행 ID |
| --- | --- |
| `GT01` | `SRC-B2-CH03-TOY-CBOW` |
| `GT02` | `SRC-B2-CH04-PTB-CBOW`, `SRC-B2-CH04-PTB-SKIPGRAM`; full-softmax 조건은 새 확장 |
| `GT03` | `SRC-B2-CH05-SMALL-RNNLM`, `SRC-B2-CH05-SMALL-RNNLM-CUSTOM` |
| `GT04` | `SRC-B2-CH06-LSTM-RNNLM` |
| `GT05` | `SRC-B2-CH06-BETTER-RNNLM`; RNN/LSTM recipe 조건은 새 확장 |
| `GT06` | `SRC-B2-CH07-ADDITION-SEQ2SEQ` |
| `GT07` | `SRC-B2-CH08-DATE-SEQ2SEQ` |
| `GO01` | `SRC-B2-CH08-ATTENTION-ALIGNMENT` |
