# e03. RNNLM 비교

## 원본

『밑바닥부터 시작하는 딥러닝 2』 원본 저장소 [`deep-learning-from-scratch-2`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/)의 vanilla RNNLM [`ch05/simple_rnnlm.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch05/simple_rnnlm.py)·[`ch05/train.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch05/train.py), LSTM RNNLM [`ch06/rnnlm.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch06/rnnlm.py)·[`ch06/train_rnnlm.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch06/train_rnnlm.py), BetterRnnlm [`ch06/better_rnnlm.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch06/better_rnnlm.py)·[`ch06/train_better_rnnlm.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch06/train_better_rnnlm.py)를 비교한다. 세 recipe를 공통 보고 형식으로 묶는 것은 이 도메인의 확장이다.

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e03` |
| 데이터·태스크 | PTB next-token language modeling |
| 실험 목적 | vanilla RNN, LSTM, BetterRnnlm의 perplexity와 비용 차이를 한 흐름에서 비교한다. |
| 사전 가설 | LSTM은 vanilla RNN보다 낮은 PPL을 보이고 BetterRnnlm은 가장 낮은 PPL을 달성한다. |
| 독립변인 | vanilla RNN, 1-layer LSTM, BetterRnnlm |
| 고정변수 | PTB vocabulary와 split, token-level PPL 산출법; 모델별 공식 recipe는 독립변인에 포함 |
| 종속변인·관찰값 | train/valid/test PPL, PPL AUC, gradient norm, 학습시간, parameter, peak memory, LR schedule |

## 2. 분석 계획

RNN-LSTM은 cell 비교로, LSTM-Better는 전체 recipe 비교로 나누어 해석한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

세 조건 curve가 반복 실험에서 분리되고 BetterRnnlm test PPL이 `75.76 +/- 3` 범위를 만족한다.

## 4. 조회할 원자 실행

`LM-RNN-C025`, `LM-LSTM-C025`, `LM-BETTER`

## 5. 해석 제한

BetterRnnlm과 기본 LSTM 차이를 개별 구성요소의 인과효과로 해석하지 않는다.
