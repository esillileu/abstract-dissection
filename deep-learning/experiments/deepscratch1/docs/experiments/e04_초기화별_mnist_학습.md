# e04. 초기화별 MNIST 학습

<!-- Domain: deepscratch1 -->

## 원본

『밑바닥부터 시작하는 딥러닝』 1권 원본 저장소 [`WegraLee-deep-learning-from-scratch`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/)의 [`ch06/weight_init_compare.py`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch06/weight_init_compare.py)를 바탕으로 한다. 원본의 MNIST MLP 초기화 비교를 유지하되, 조건을 명시적으로 분리하고 10-seed curve·AUC 분석을 추가했다.

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e04` |
| 데이터·태스크 | MNIST 전체 학습 세트의 10-class 분류 |
| 실험 목적 | activation 분포 차이가 실제 학습 수렴성 차이로 이어지는지 확인한다. |
| 사전 가설 | ReLU MLP에서 He와 Xavier는 정상 수렴하며 std=.01은 학습 지연 또는 정체를 보인다. |
| 독립변인 | Normal std=.01, Xavier, He |
| 고정변수 | `784-[100x4]-10`, ReLU, SGD lr=.01, batch 128, 2,000 updates |
| 종속변인·관찰값 | loss curve, normalized AUC, 목표 loss 도달 step, accuracy, 실패율, 최종 층별 activation 분포 |

## 2. 분석 계획

동일 seed paired loss curve와 AUC를 비교하고 activation probe 결과와 연결한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

He·Xavier는 finite하게 수렴한다. std=.01은 He보다 loss AUC가 높고 2,000-step loss가 높은 seed가 8/10 이상이다.

## 4. 조회할 원자 실행

`MLP-SGD-STD001`, `MLP-SGD-XAVIER`, `MLP-SGD-HE`

## 5. 해석 제한

`MLP-SGD-HE`는 e02 결과를 재사용한다.
