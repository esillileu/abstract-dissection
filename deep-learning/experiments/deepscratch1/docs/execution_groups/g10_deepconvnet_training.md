# g10. cnn_accuracy_reproduction

<!-- Domain: deepscratch1 -->

## 공통 실행 설정

SimpleConvNet과 DeepConvNet은 각각 Adam `.001`, batch 100, replacement sampling으로 20 epochs 학습한다. SimpleConvNet은 `Conv30 5x5-ReLU-Pool-FC100-ReLU-FC10`과 `std=.01` 초기화를 사용한다. DeepConvNet은 `Conv16-Conv16-Pool-Conv32-Conv32-Pool-Conv64-Conv64-Pool-FC50-ReLU-Dropout(.5)-FC10-Dropout(.5)`와 He 초기화를 사용한다.

Convolution pad는 원본과 같이 `[1,1,1,2,1,1]`이고 stride는 모두 1이다.

## 원자 조건

`CNN-SIMPLE-ACCURACY`, `CNN-DEEP-ACCURACY`: e09가 각각 독립적으로 실행하는 원본 CNN 학습 recipe. 규정 accuracy는 실행 조건이 아니라 e09 분석 기준이다.

## 사용 실험

e09
