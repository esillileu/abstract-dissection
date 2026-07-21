# e02. Optimizer MNIST 학습

<!-- Domain: deepscratch1 -->

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e02` |
| 데이터·태스크 | MNIST 전체 학습 세트의 10-class 분류 |
| 실험 목적 | 동일한 MLP와 학습 예산에서 optimizer별 학습 진도, 안정성, 최종 성능을 비교한다. |
| 사전 가설 | Momentum, AdaGrad, Adam 중 다수는 SGD보다 낮은 loss AUC와 빠른 목표 loss 도달을 보인다. |
| 독립변인 | SGD, Momentum, AdaGrad, Adam |
| 고정변수 | `784-[100x4]-10`, ReLU, He, batch 128, 2,000 updates |
| 종속변인·관찰값 | iter-loss curve, normalized AUC, 목표 loss 도달 step, train/test accuracy, 실패율, 가능하면 최적화 궤적 대리지표 |

## 2. 분석 계획

동일 seed의 paired curve와 AUC 차이를 분석하고 마지막 100-step smoothed loss를 비교한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

모든 조건 finite loss. optimizer 간 normalized AUC 분포 차이가 재현되고, 가속형 optimizer 중 적어도 2개가 SGD보다 우세한 seed가 8/10 이상이다.

## 4. 조회할 원자 실행

`MLP-SGD-HE`, `MLP-MOM-HE`, `MLP-ADAGRAD-HE`, `MLP-ADAM-HE`

## 5. 해석 제한

최종 정확도만으로 optimizer 우열을 판정하지 않는다.
