# g10. deepconvnet_training

<!-- Domain: deepscratch1 -->

## 공통 실행 설정

`Conv16-Conv16-Pool-Conv32-Conv32-Pool-Conv64-Conv64-Pool-FC50-ReLU-Dropout(.5)-FC10-Dropout(.5)`; He; Adam `.001`; batch 100; replacement sampling; 2 epochs.

Convolution pad는 원본과 같이 `[1,1,1,2,1,1]`이고 stride는 모두 1이다.

## 원자 조건

`CNN-DEEP-ACCURACY`: 원본 DeepConvNet 학습 recipe. 규정 accuracy는 실행 조건이 아니라 e09 분석 기준이다.

## 사용 실험

e09
