# e04. Seq2seq 입력 반전과 구조 비교

## 원본

원본 저장소 [`deep-learning-from-scratch-2`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/)의 [`ch07/seq2seq.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch07/seq2seq.py), [`ch07/peeky_seq2seq.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch07/peeky_seq2seq.py), [`ch07/train_seq2seq.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch07/train_seq2seq.py)를 기반으로 한다. Attention 조건은 [`ch08/attention_seq2seq.py`](../../../../01_deep-learning-from-base/deep-learning-from-scratch-2/ch08/attention_seq2seq.py)를 같은 addition task에 적용한 확장이고, forward/reverse의 전 요인 비교도 이 도메인에서 추가했다.

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e04` |
| 데이터·태스크 | `addition.txt` character-level 덧셈 답 생성 |
| 실험 목적 | 입력 반전과 Peeky·Attention 구조가 exact-match 학습에 미치는 영향을 한 실험에서 비교한다. |
| 사전 가설 | reverse가 forward보다 유리하고, Peeky와 Attention은 vanilla보다 context 전달을 개선한다. |
| 독립변인 | Vanilla forward/reverse, Peeky forward/reverse, Attention reverse |
| 고정변수 | embedding 16, hidden 128, batch 128, 25 epochs, Adam .001, clip 5, greedy decode |
| 종속변인·관찰값 | exact-match, token accuracy, accuracy AUC, 목표 accuracy 도달 epoch, 수렴속도, parameter, 시간, memory, attention map |

## 2. 분석 계획

먼저 같은 구조 내 forward-reverse를 비교하고, 이후 reverse 조건끼리 Vanilla-Peeky-Attention을 비교한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

reverse 조건이 같은 구조의 forward보다 우세하고, 구조 간 curve 분포 차이가 반복 실험에서 확인된다.

## 4. 조회할 원자 실행

`SEQA-VAN-FWD`, `SEQA-VAN-REV`, `SEQA-PEEKY-FWD`, `SEQA-PEEKY-REV`, `SEQA-ATTN-REV`

## 5. 해석 제한

입력 반전 효과와 architecture 효과를 하나의 단일 대비로 섞지 않고 두 단계 대비로 분석한다.
