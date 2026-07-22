# e09. SimpleConvNet·DeepConvNet 규정 성능 재현

<!-- Domain: deepscratch1 -->

## 원본

『밑바닥부터 시작하는 딥러닝』 1권 원본 저장소 [`WegraLee-deep-learning-from-scratch`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/)의 [`ch07/simple_convnet.py`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch07/simple_convnet.py)·[`ch07/train_convnet.py`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch07/train_convnet.py), 그리고 [`ch08/deep_convnet.py`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch08/deep_convnet.py)·[`ch08/train_deepnet.py`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch08/train_deepnet.py)를 직접 재현한다. 전체 test-set 평가와 10-seed 보고만 원본의 1,000개 표본 평가를 확장한다.

## 실험 정의

| 항목 | 내용 |
| --- | --- |
| 데이터·태스크 | 공식 MNIST train 60,000 / test 10,000, 이미지 10-class 분류 |
| 실험 목적 | 책의 SimpleConvNet과 DeepConvNet 구조·optimizer recipe를 사용해, 각 20-epoch 학습에서 규정 test accuracy 도달 여부와 학습 곡선을 확인한다. |
| 독립변인 | architecture recipe: SimpleConvNet, DeepConvNet |
| 고정변수 | Adam `lr=.001`, batch 100, replacement sampling, 20 epochs, float64 CUDA |
| 규정 성능 | SimpleConvNet final full-test accuracy `>=98.96%`; DeepConvNet final full-test accuracy `>=99.38%` |
| 주 분석 | epoch별 full-test accuracy graph |

## 원본 recipe

- SimpleConvNet: 원본 `ch07/train_convnet.py`와 `ch07/simple_convnet.py`의 `Conv30 5×5 → ReLU → Pool → FC100 → ReLU → FC10`, `std=.01` 초기화.
- DeepConvNet: 원본 `ch08/train_deepnet.py`와 `ch08/deep_convnet.py`의 6-convolution recipe, pad `[1,1,1,2,1,1]`, FC50, dropout `.5`, He 초기화.
- 두 원본 학습 스크립트 모두 Adam `.001`, mini-batch 100, 20 epochs다.

원본은 매 epoch train/test 앞 1,000개 표본을 평가한다. e09는 규정 성능을 명확히 하기 위해 동일 학습 recipe를 유지하되, graph와 최종 기준에는 전체 test 10,000개 accuracy를 사용한다.

학습 중간에는 첫 update와 이후 매 20 updates마다 고정 validation 1,000개와 별도 고정 train 1,000개 probe의 accuracy를 기록한다. Train probe 기록은 `training.record_step_train_evaluation`으로 켜며, 이 옵션의 기본값은 `false`다.

## 분석과 보고

- SimpleConvNet·DeepConvNet의 full-test accuracy curve를 한 그래프에 표시한다.
- 각 curve는 10개 고정 seed의 평균과 최솟값·최댓값을 표시한다.
- 각 seed의 final test accuracy와 해당 모델의 규정 성능 도달 여부를 CSV로 기록한다.
- 일반 성능 비교는 curve의 수렴 양상과 final test accuracy를 기술하되, 두 구조의 차이를 depth 하나의 인과효과로 단정하지 않는다.

## 실행 그룹

e09는 다른 실험 ID를 참조하지 않는다. SimpleConvNet과 DeepConvNet 모두 e09가 소유한 `g10` 실행 그룹에서 독립적으로 학습한다. 분석은 MLflow의 `execution_group.id` 태그로 해당 결과를 조회한다.

## 원자 실행

`g10/CNN-SIMPLE-ACCURACY`, `g10/CNN-DEEP-ACCURACY`
