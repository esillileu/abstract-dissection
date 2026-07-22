# e07. RNNLM small-corpus comparison

ch05의 Vanilla RNNLM 조건을 공통 baseline으로 하여 RNN, LSTM, BetterRnnlm을 비교한다.

| 항목 | 내용 |
|---|---|
| 데이터 | PTB train의 첫 1,000 tokens |
| 독립변인 | Vanilla RNN, LSTM, BetterRnnlm |
| 고정변수 | embedding/hidden 100, batch 10, BPTT 5, SGD .1, 100 epochs, gradient clipping 없음 |
| 평가 | interval train PPL, 학습 종료 후 PTB test PPL 1회 |

## 조회할 원자 실행

`LM-TOY-RNN`, `LM-TOY-LSTM`, `LM-TOY-BETTER`

## 해석 제한

Vanilla RNN은 ch05 책 조건이다. LSTM과 BetterRnnlm은 같은 작은 corpus와 학습 recipe에 맞춘 통제 확장이다.
