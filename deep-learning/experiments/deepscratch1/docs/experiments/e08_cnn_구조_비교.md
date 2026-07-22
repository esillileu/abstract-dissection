# e08. 공간적 배치 활용 비교

<!-- Domain: deepscratch1 -->

## 원본

『밑바닥부터 시작하는 딥러닝』 1권 원본 저장소 [`WegraLee-deep-learning-from-scratch`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/)의 [`ch07/simple_convnet.py`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch07/simple_convnet.py)와 [`ch07/train_convnet.py`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch07/train_convnet.py)의 SimpleConvNet recipe를 사용한다. 고정 픽셀 순열 및 parameter-matched MLP 대조군은 CNN의 공간 배치 의존성을 검증하기 위해 이 도메인에서 추가한 통제 확장이다.

## 1. 실험 정의

| 항목            | 내용                                                                                                                                                       |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 실험 ID         | `e08`                                                                                                                                                      |
| 데이터·태스크   | MNIST 이미지 10-class 분류                                                                                                                                 |
| 실험 목적       | 원본 이미지와 고정 픽셀 순열 이미지에서 ParameterMatchedNN과 SimpleConvNet의 학습 성능을 비교하여 CNN이 이미지의 공간적 배치를 학습에 활용하는지 확인한다. |
| 사전 가설       | 고정 픽셀 순열 적용 시 SimpleConvNet의 accuracy와 학습 효율 감소 폭이 ParameterMatchedNN보다 크게 나타난다.                                                |
| 독립변인        | 모델 구조: ParameterMatchedNN, SimpleConvNet / 입력 조건: 원본, 고정 픽셀 순열                                                                             |
| 고정변수        | MNIST split, 약 433.9K parameters, 학습 가능 계층 3개, Adam lr=.001, batch 100, 2 epochs, full-test epoch 평가, replacement sampling               |
| 종속변인·관찰값 | epoch별 full-test accuracy curve, 마지막 epoch test accuracy의 paired permutation drop                                                                     |

## 2. 비교 모델

| 모델               | 구조                                 | Parameter count |
| ------------------ | ------------------------------------ | --------------: |
| ParameterMatchedNN | Flatten-FC489-ReLU-FC100-ReLU-FC10   |         433,875 |
| SimpleConvNet      | Conv30 5x5-ReLU-Pool-FC100-ReLU-FC10 |         433,890 |

두 모델의 parameter count 차이는 15개이며, SimpleConvNet 대비 약 0.0035%이다.

구현은 책의 `ch07/simple_convnet.py`를 따르는 `SimpleCNN`과, 책의 `common/multi_layer_net.py`를 일반화한 기존 `MLP`를 사용한다. ParameterMatchedNN은 후자의 구조값만 `784-489-100-10`으로 고정한 조건이며, 별도 구현을 추가하지 않는다.

## 3. 분석 계획

주 분석은 모델별 원본·순열 full-test accuracy curve다. 각 curve는 10개 고정 seed의 평균과 seed 간 최솟값·최댓값을 함께 표시한다.

보조 요약으로 마지막 epoch에서 동일 master seed의 원본·순열 차이를 계산한다. 이는 curve를 대체하는 단일 성능 순위 지표가 아니라, curve에서 관찰한 차이를 seed 단위로 확인하기 위한 표다.

```text
permutation_drop
    = metric_original - metric_permuted
```

절대 accuracy 차이는 전체 architecture 차이로 해석하며, 주요 분석은 동일 모델 내부의 원본·순열 차이와 두 모델의 감소량 차이를 대상으로 한다.

### 필수 보고물

- ParameterMatchedNN의 원본·순열 test accuracy curve
- SimpleConvNet의 원본·순열 test accuracy curve
- 각 조건의 seed별 원자료와 마지막 epoch test accuracy 요약표
- 동일 seed의 마지막 epoch `permutation_drop` 표
- 실패 run 유무

## 4. 재현·달성 기준

`CNN-SIMPLE`의 원본 MNIST test accuracy가 기존 약 98.96% 범위를 반복 실험 편차 내에서 재현되고, 네 원자 조건의 매 seed에서 원본·순열 curve와 마지막 epoch 성능 차이를 계산할 수 있어야 한다.

사전 가설과 다른 결과가 나타나더라도 모든 조건의 결과가 확보되면 실험은 완료된 것으로 판단한다.

## 5. 조회할 원자 실행

`NN-MATCHED`, `NN-MATCHED-PERMUTED`, `CNN-SIMPLE`, `CNN-SIMPLE-PERMUTED`

## 6. 해석 제한

ParameterMatchedNN과 SimpleConvNet은 parameter count와 계층 수를 맞추었지만 연산 방식은 다르므로, 절대 성능 차이를 국소 연결이나 가중치 공유의 독립적 효과로 해석하지 않는다.

SimpleConvNet의 permutation drop이 ParameterMatchedNN보다 크게 나타날 경우, 현재 MNIST 조건에서 SimpleConvNet이 픽셀의 공간적 배치 정보에 더 크게 의존한 것으로 해석한다.
