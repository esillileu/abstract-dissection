# 1권 원본 시행(trial) 기록 레지스트리

대상 원본은 [WegraLee-deep-learning-from-scratch](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/)다. 이 문서는 실험 장 단위가 아니라 **고정된 데이터·모델·파라미터·학습 절차로 한 번 실행하는 시행**을 정리한다.

ch06–ch08을 하이퍼파라미터 기반 실행 그룹으로 재구성한 목록은 [book1_ch06plus_execution_groups.md](book1_ch06plus_execution_groups.md)에 있다.

- 분석용 평균/표준편차, AUC, threshold 도달, paired difference, 최종 모델 순위는 포함하지 않는다.
- 구현·데모·단순 표시 예제는 제외했다. 아래의 값은 원본이 실제 학습 중 또는 평가에서 기록·출력·그린 raw 값이다.
- 표의 range ID를 전개하면 149개의 source-default condition이다. 재구성의 실제 run ID는 `<trial_id>.seed-<n>`이다.

## 공통 raw record

| 원본의 기록 방식 | 보존할 raw record |
| --- | --- |
| 매 update append한 loss | `update`, `epoch`, `train/objective`, `batch_size` |
| epoch마다 평가한 accuracy | `update`, `epoch`, `split`, `evaluation_set_id`, `example_count`, `accuracy` |
| 목적함수 위 궤적 | `update`, 상태 좌표, objective, gradient |
| hyperparameter search | 시행 ID, resolved `lr`·weight decay, epoch별 train/valid accuracy |

`mnist-train-first-1000`, `mnist-test-first-1000`, `mnist-test-full`처럼 평가 slice ID를 명시해 원본의 부분 평가와 전체 평가를 구별한다.

## 시행

| 시행 ID | 원본 실행·고정 조건 | 원본이 기록한 값과 시점 |
| --- | --- | --- |
| `dlfs1.ch04.mlp-sgd` | [`ch04/train_neuralnet.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch04/train_neuralnet.py); MNIST full train/test, `784-50-10`, SGD `.1`, batch `100`, `10,000` updates | 매 update mini-batch cross-entropy loss; 매 epoch 시작(`i % 600 == 0`) full train/test accuracy를 list·콘솔에 기록 |
| `dlfs1.ch05.mlp-backprop-sgd` | [`ch05/train_neuralnet.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch05/train_neuralnet.py); MNIST full train/test, `784-50-10`, backprop SGD `.1`, batch `100`, `10,000` updates | 매 update mini-batch cross-entropy loss; 매 epoch 시작 full train/test accuracy를 콘솔·list에 기록 |
| `dlfs1.ch06.optimizer-path.sgd` | [`ch06/optimizer_compare_naive.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch06/optimizer_compare_naive.py); `f=x²/20+y²`, init `(-7,2)`, SGD lr `.95`, `30` updates | 각 update **전** `x`, `y`를 append. objective·gradient는 같은 좌표에서 계산해 함께 보존 |
| `dlfs1.ch06.optimizer-path.momentum` | 같은 source; Momentum lr `.1`, momentum `.9`, `30` updates | 각 update 전 `x`, `y` 궤적 |
| `dlfs1.ch06.optimizer-path.adagrad` | 같은 source; AdaGrad lr `1.5`, `30` updates | 각 update 전 `x`, `y` 궤적 |
| `dlfs1.ch06.optimizer-path.adam` | 같은 source; Adam lr `.3`, `30` updates | 각 update 전 `x`, `y` 궤적 |
| `dlfs1.ch06.optimizer-mnist.sgd` | [`ch06/optimizer_compare_mnist.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch06/optimizer_compare_mnist.py); MNIST full train, `784-[100×4]-10`, batch `128`, `2,000` updates, SGD lr `.01` | 매 update mini-batch loss. 원본의 smoothing 전 raw loss를 보존 |
| `dlfs1.ch06.optimizer-mnist.momentum` | 같은 source·data·model·budget; Momentum lr `.01`, momentum `.9` | 매 update mini-batch loss |
| `dlfs1.ch06.optimizer-mnist.adagrad` | 같은 source·data·model·budget; AdaGrad lr `.01` | 매 update mini-batch loss |
| `dlfs1.ch06.optimizer-mnist.adam` | 같은 source·data·model·budget; Adam lr `.001` | 매 update mini-batch loss |
| `dlfs1.ch06.init-compare.std-001` | [`ch06/weight_init_compare.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch06/weight_init_compare.py); MNIST full train, `784-[100×4]-10`, SGD `.01`, batch `128`, `2,000` updates, Normal std `.01` | 매 update mini-batch loss; 100 update마다 같은 batch loss를 콘솔에 출력 |
| `dlfs1.ch06.init-compare.xavier` | 같은 source; initializer `sqrt(1/fan_in)` | 매 update mini-batch loss; 100 update마다 console loss |
| `dlfs1.ch06.init-compare.he` | 같은 source; initializer `sqrt(2/fan_in)` | 매 update mini-batch loss; 100 update마다 console loss |
| `dlfs1.ch06.weight-decay.lambda-01` | [`ch06/overfit_weight_decay.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch06/overfit_weight_decay.py); MNIST train 첫 `300`, official test, `784-[100×6]-10`, SGD `.01`, batch `100`, L2 `.1`, `201` epoch observations | update `1,4,7,…,601` 뒤 full 300-train 및 full test accuracy. loss는 선언되지만 append하지 않음 |
| `dlfs1.ch06.dropout.on-ratio-02` | [`ch06/overfit_dropout.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch06/overfit_dropout.py); MNIST train 첫 `300`, official test, `784-[100×6]-10`, SGD `.01`, batch `100`, dropout on `.2`, `301` epochs | 매 update loss; 매 epoch 시작 full 300-train/test accuracy |
| `dlfs1.ch06.batchnorm.scale-01.bn-off` … `dlfs1.ch06.batchnorm.scale-16.bn-off` | [`ch06/batch_norm_test.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch06/batch_norm_test.py); MNIST train 첫 `1,000`, `784-[100×5]-10`, SGD `.01`, batch `100`, `20` epoch observations, BN off, `scale-k = logspace(0,-4,16)[k-1]` | 각 scale은 독립 시행. update `1,11,21,…,191` 뒤 full 1,000-train accuracy만 기록 |
| `dlfs1.ch06.batchnorm.scale-01.bn-on` … `dlfs1.ch06.batchnorm.scale-16.bn-on` | 위와 동일하되 BatchNorm on | 같은 update 위치의 full 1,000-train accuracy. source가 같은 process에서 실행해도 model condition별 독립 ID로 저장 |
| `dlfs1.ch06.hp-search.trial-001` … `dlfs1.ch06.hp-search.trial-100` | [`ch06/hyperparameter_optimization.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch06/hyperparameter_optimization.py); MNIST 처음 `500` shuffle 후 100 valid/400 train, `784-[100×6]-10`, SGD, batch `100`, `50` epochs. `lr=10^U(-6,-2)`, `weight_decay=10^U(-8,-4)` | 각 시행의 resolved `lr`·weight decay와 epoch별 train/valid accuracy. 원본의 마지막 valid accuracy 정렬은 분석 산출물 |
| `dlfs1.ch07.simple-convnet` | [`ch07/train_convnet.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch07/train_convnet.py); MNIST image full train/test, Conv30 5×5→FC100, Adam `.001`, batch `100`, `20` epochs, first-`1,000` eval | 매 update loss; 매 epoch 시작 first-1,000 train/test accuracy; 종료 후 full test accuracy와 `params.pkl` checkpoint |
| `dlfs1.ch08.deep-convnet` | [`ch08/train_deepnet.py`](01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch08/train_deepnet.py); MNIST image full train/test, DeepConvNet, Adam `.001`, batch `100`, `20` epochs, first-`1,000` eval | 매 update loss; 매 epoch 시작 first-1,000 train/test accuracy; 종료 후 full test accuracy와 `deep_convnet_params.pkl` checkpoint |

## source 내 수동 변경값

`weight_init_activation_histogram.py`의 activation·scale 주석 전환, `overfit_dropout.py`의 on/off·ratio 전환, `overfit_weight_decay.py`의 lambda 전환은 snapshot에서 각각 한 값만 실행되도록 되어 있다. 이 문서는 source-default condition만 포함한다. 변형을 재구성 registry에 넣을 때는 기존 ID를 재사용하지 않고 새 condition ID를 발급한다.
