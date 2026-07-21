# deepscratch1 문서

이 디렉터리는 『밑바닥부터 시작하는 딥러닝』 1권을 재현하는 `deepscratch1` 실험 도메인의 설계·실행 문서를 소유한다. 실행 설정은 인접한 [`../config/`](../config/)에, 분석 코드는 [`../analysis/`](../analysis/)에 있다.

- [책 원본 대비 차이](difference.md)

## 실험 설계

- [e01 Optimizer 목적함수 경로](experiments/e01_optimizer_toy_목적함수_경로.md)
- [e02 Optimizer MNIST 학습](experiments/e02_optimizer_mnist_학습.md)
- [e03 초기화별 activation 분포](experiments/e03_초기화별_activation_분포.md)
- [e04 초기화별 MNIST 학습](experiments/e04_초기화별_mnist_학습.md)
- [e05 BatchNorm-초기화 scale](experiments/e05_batchnorm_초기화_scale.md)
- [e06 Weight decay](experiments/e06_weight_decay.md)
- [e07 Dropout](experiments/e07_dropout.md)
- [e08 공간적 배치 활용](experiments/e08_cnn_구조_비교.md)
- [e09 SimpleConvNet·DeepConvNet 규정 성능 재현](experiments/e09_cnn_규정성능_재현.md)

## 실행 그룹

원자 실행의 구조·공통 정책은 [`execution_groups/`](execution_groups/)에 있다. 각 문서는 YAML의 `execution_group_id`와 대응한다.

상위 `experiments/docs/`에는 여러 도메인이 공유하는 MLflow 스키마·런타임 명세와, 아직 도메인으로 이전되지 않은 문서만 둔다.
