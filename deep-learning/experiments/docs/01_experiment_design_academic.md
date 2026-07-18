# 딥러닝 밑바닥 실험 재현: 상위 수준 실험 설계서

> 관점: 학술적 재현 및 ML 실험 설계
> 범위: 『밑바닥부터 시작하는 딥러닝』 1·2권의 선정 실험 17개
> 실행 명세: 별도 문서 `02_atomic_run_execution_spec.md` 참조

## 1. 문서 목적

이 문서는 각 실험이 **어떤 연구 질문에 답하는지**, 무엇을 조작하고 무엇을 측정하며, 어떤 분석으로 목적을 확인할지 정의한다. 모델 구현·배치 크기·학습률·체크포인트 정책과 같은 실행 세부사항은 시행 명세서로 분리한다.

실험 문서는 실행 결과를 소유하지 않는다. 각 실험은 시행 명세서의 원자 시행(atomic run)들을 조회해 분석한다.

## 2. 공통 연구 방법

### 2.1 반복 단위

- 원자 시행의 seed 제외 조건을 MLflow parent run으로 둔다.
- 확률적 시행은 parent당 child seed `0..9`의 10회 반복을 수행한다.
- 데이터 split은 child seed와 분리해 고정한다.
- 비교 조건 사이에서 같은 seed는 가능한 한 초기 weight 원본, minibatch 순서, dropout·sampling 난수열을 대응시킨다.
- 결정적인 e01 canonical 재현은 반복하지 않는다.

### 2.2 통계 보고

- 모든 주요 지표에 seed별 원자료, 평균, 표준편차, median, 95% 신뢰구간을 보고한다.
- 비교의 기본 단위는 같은 seed 사이의 paired difference다.
- 작은 표본 수를 고려해 paired difference의 bootstrap 95% CI를 주 추론값으로 사용하고 paired t-CI를 함께 보존한다.
- 곡선 비교는 최종값만 보지 않고 normalized AUC, time-to-threshold, 실패율을 함께 사용한다.
- 실패·NaN·발산 run을 자동 제외하지 않는다. 완료 run 통계와 실패율을 함께 보고한다.
- 다수 조건 sweep에서 사후 pairwise 검정을 수행할 경우 Holm 보정을 적용한다.

### 2.3 재현 기준의 종류

- **정확 재현**: 결정적 코드의 좌표·출력값이 수치 허용오차 안에서 일치한다.
- **수치 재현**: 공개 구현의 명시적 목표 또는 널리 알려진 기준 범위에 도달한다.
- **방향 재현**: 조건 간 순위, 곡선 형태, 안정성 또는 민감도 패턴이 재현된다.
- **운영 기준**: 공개 코드에 단일 목표값이 없을 때 본 프로젝트가 사전에 정의한 acceptance threshold다. 원전의 공식 수치로 오해하지 않는다.

## 3. 실험 요약

| ID | 실험 | 데이터·태스크 | 핵심 독립변인 | 분석 시행 세트 |
| --- | --- | --- | --- | --- |
| e01 | Optimizer toy 목적함수 경로 비교 | 2차원 연속 최적화 | optimizer | TOY-SGD, TOY-MOM, TOY-ADAGRAD, TOY-ADAM |
| e02 | Optimizer MNIST 학습 비교 | MNIST 다중분류 | optimizer | MLP-SGD-HE, MLP-MOM-HE, MLP-ADAGRAD-HE, MLP-ADAM-HE |
| e03 | 초기화에 따른 activation 분포 | 합성 입력 forward 진단 | activation × initializer | ACT-* 12개 |
| e04 | 초기화 MNIST 학습 비교 | MNIST 다중분류 | initializer | MLP-SGD-STD001, MLP-SGD-XAVIER, MLP-SGD-HE |
| e05 | BatchNorm과 초기화 scale 민감도 | MNIST 분류·학습 안정성 | BatchNorm × weight scale | BN-OFF-01..16, BN-ON-01..16 |
| e06 | Weight decay 과적합 억제 | 저데이터 MNIST 분류 | weight decay λ | REG-BASE, REG-WD-* |
| e07 | Dropout 과적합 억제 | 저데이터 MNIST 분류 | dropout ratio | REG-BASE, REG-DO-* |
| e08 | CNN 구조 비교 | MNIST 이미지 분류 | architecture recipe | CNN-SIMPLE, CNN-DEEP |
| e09 | 추론 정밀도 비교 | MNIST 추론 평가 | inference dtype | DTYPE-F64, DTYPE-F32, DTYPE-F16 |
| e10 | CBOW vs Skip-gram | PTB 단어 임베딩 | architecture | W2V-CBOW-NS, W2V-SG-NS |
| e11 | Full softmax vs Negative Sampling | PTB 단어 임베딩 | objective | W2V-CBOW-FULL, W2V-CBOW-NS |
| e12 | Simple RNN vs LSTM | PTB 다음 단어 예측 | recurrent cell | LM-RNN-C025, LM-LSTM-C025 |
| e13 | Gradient clipping threshold | PTB LSTM 언어모델 | max_grad | LM-LSTM-NONE, LM-LSTM-C010..C500 |
| e14 | RNNLM vs BetterRnnlm | PTB 다음 단어 예측 | model/training recipe | LM-LSTM-C025, LM-BETTER |
| e15 | Seq2seq 입력 반전 | 덧셈 문자열 생성 | reverse | SEQA-VAN-FWD, SEQA-VAN-REV |
| e16 | Seq2seq vs Peeky Seq2seq | 덧셈 문자열 생성 | architecture | SEQA-VAN-REV, SEQA-PEEKY-REV |
| e17 | Attention Seq2seq | 날짜 형식 변환 생성 | architecture | SEQD-VAN-REV, SEQD-PEEKY-REV, SEQD-ATTN-REV |

## 4. 실험별 설계

### e01. Optimizer toy 목적함수 경로 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | 실제 데이터셋 없음. 목적함수 `f(x,y)=x^2/20+y^2`의 2차원 연속 최적화. |
| 실험 목적 | 비등방성 곡면에서 optimizer의 축별 진동, 이동 경로, 초기 수렴 및 최종 접근 특성을 재현한다. |
| 사전 가설 | Momentum·AdaGrad·Adam은 SGD와 다른 경로 및 진동 양상을 보이며, 각 optimizer의 공식 설정에서 공개 예제와 동일한 30-step 궤적을 생성한다. |
| 독립변인 | optimizer 4수준: SGD(`lr=.95`), Momentum(`lr=.1, momentum=.9`), AdaGrad(`lr=1.5`), Adam(`lr=.3, β1=.9, β2=.999`). |
| 고정변수 | 목적함수, 해석적 gradient, 초기점 `(-7,2)`, update 30회, float64. canonical 재현은 결정적이므로 1회만 시행한다. |
| 종속변인 | step별 `(x,y)`, 목적함수 값, 최적점까지 거리, 경로 길이, x/y축 부호 변경 횟수. |
| 목적 확인을 위한 분석 | 공개 코드의 기준 궤적과 step별 좌표를 비교한다. 보조적으로 optimizer별 경로와 등고선을 한 그림에 제시한다. |
| 재현 기준 | 정확 재현. 30번째 update 후 목적함수: SGD `0.0133268`, Momentum `0.0479370`, AdaGrad `0.000721949`, Adam `0.102205`; 절대오차 `<=1e-6`. 경로 좌표도 동일 허용오차를 만족해야 한다. |
| 분석 시행 세트 | `TOY-SGD`, `TOY-MOM`, `TOY-ADAGRAD`, `TOY-ADAM` |

### e02. Optimizer MNIST 학습 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | MNIST 전체 학습 세트, 정규화·flatten. 10-class 이미지 분류. |
| 실험 목적 | 동일한 MLP와 학습 예산에서 optimizer별 학습 속도, 안정성 및 최종 성능 차이를 재현한다. |
| 사전 가설 | Momentum·AdaGrad·Adam 중 적어도 둘은 SGD보다 초기·중기 loss 감소가 빠르며 loss AUC가 낮다. |
| 독립변인 | optimizer 4수준: SGD, Momentum, AdaGrad, Adam. 각 optimizer의 학습률 및 모멘텀 계수는 시행 명세 문서에 고정한다. |
| 고정변수 | MLP `784-[100×4]-10`, ReLU, He 초기화, batch 128, 2,000 updates, 동일 child seed에서 초기 파라미터와 minibatch index 열을 공유. |
| 종속변인 | step별 train loss, smoothed loss, loss AUC, 목표 loss 도달 step, final train/test accuracy, 실패율, wall time. |
| 목적 확인을 위한 분석 | seed별 paired curve를 정렬해 평균과 95% CI를 그린다. 주 분석은 normalized loss AUC와 `time_to_loss`이며, 최종 정확도는 보조 지표다. |
| 재현 기준 | 방향 재현. 모든 조건에서 NaN·발산 0/10. 마지막 100-step median loss가 첫 100-step보다 80% 이상 감소. Momentum·AdaGrad·Adam 중 2개 이상이 SGD보다 paired loss AUC가 낮은 seed를 8/10 이상 확보. |
| 분석 시행 세트 | `MLP-SGD-HE`, `MLP-MOM-HE`, `MLP-ADAGRAD-HE`, `MLP-ADAM-HE` |

### e03. 가중치 초기화에 따른 activation 분포 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | 표준정규 합성 입력 `(1000,100)`. 학습 없는 5-layer forward signal propagation 진단. |
| 실험 목적 | activation과 초기화 조합이 깊이에 따라 분산 소실·폭주·포화·dead activation을 유발하는 양상을 재현한다. |
| 사전 가설 | 작은 고정 scale은 분산을 소실시키고, 큰 scale의 sigmoid는 포화를 유발한다. Xavier는 sigmoid/tanh, He는 ReLU에서 상대적으로 안정적인 분포를 유지한다. |
| 독립변인 | activation `{sigmoid,tanh,relu}` × initializer `{std=1,std=.01,Xavier,He}`의 3×4 완전요인 설계. |
| 고정변수 | 입력 분포·표본수, width 100, hidden depth 5, bias 0. 같은 seed에서는 입력과 표준정규 weight 원본을 공유하고 scale만 변경. |
| 종속변인 | 층별 mean/std/percentile, zero ratio, saturation ratio, 첫 층 대비 마지막 층 std 비율, histogram. |
| 목적 확인을 위한 분석 | 각 조합의 층별 분포와 signal-retention curve를 비교한다. activation별로 initializer 효과를 보고, initializer별로 activation 상호작용을 확인한다. |
| 재현 기준 | 패턴 재현. `std=.01`은 깊이에 따라 activation std가 감소해야 한다. `sigmoid/std=1`은 포화 비율이 증가해야 한다. `sigmoid·tanh/Xavier`와 `ReLU/He`는 대응하지 않는 극단 조건보다 마지막/첫 층 std 비율이 1에 가깝고 수치 실패가 없어야 한다. |
| 분석 시행 세트 | `ACT-*` 12개 전부 |

### e04. 가중치 초기화 MNIST 학습 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | MNIST 전체 학습 세트. 10-class 분류. |
| 실험 목적 | forward 분포 차이가 실제 최적화 가능성과 수렴 속도로 이어지는지 검증한다. |
| 사전 가설 | ReLU MLP에서 He와 Xavier는 정상적으로 수렴하지만 `std=.01`은 gradient와 activation이 작아져 학습이 현저히 느리다. |
| 독립변인 | initializer 3수준: fixed normal std `.01`, Xavier, He. |
| 고정변수 | MLP `784-[100×4]-10`, ReLU, SGD `lr=.01`, batch 128, 2,000 updates, 동일 seed의 데이터 순서. |
| 종속변인 | step별 loss, normalized loss AUC, 목표 loss 도달 step, final accuracy, 수렴 실패율. |
| 목적 확인을 위한 분석 | seed별 paired loss curve와 AUC를 비교한다. He와 Xavier의 차이, 그리고 두 조건 대비 `std=.01`의 정체를 별도로 보고한다. |
| 재현 기준 | 방향 재현. He·Xavier는 NaN 없이 학습하고 최종 smoothed loss `<.5`. `std=.01`의 median loss AUC가 He보다 높고, 8/10 이상의 paired seed에서 2,000-step loss가 He보다 높아야 한다. |
| 분석 시행 세트 | `MLP-SGD-STD001`, `MLP-SGD-XAVIER`, `MLP-SGD-HE` |

### e05. Batch Normalization과 초기화 scale 민감도 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | MNIST train 앞 1,000개. 분류 및 학습 안정성 진단. |
| 실험 목적 | BatchNorm이 초기 weight scale에 대한 민감도를 줄이고 정상 학습 가능한 scale 범위를 넓히는지 검증한다. |
| 사전 가설 | BatchNorm on 조건은 no-BN보다 더 넓은 scale 구간에서 빠르고 안정적인 학습을 보인다. |
| 독립변인 | BatchNorm `{off,on}` × 초기화 scale `logspace(0,-4,16)`의 2×16 요인 설계. |
| 고정변수 | MLP `784-[100×5]-10`, ReLU, SGD `lr=.01`, batch 100, 20 epochs, 동일 scale·seed의 초기 weight와 batch 순서 공유. |
| 종속변인 | epoch별 train accuracy, final accuracy, `time_to_acc=.8`, 성공 여부, 성공 scale 개수·log 범위. |
| 목적 확인을 위한 분석 | scale별 BN on/off paired difference를 계산한다. 성공률을 scale에 대해 시각화하고 BN×scale 상호작용을 기술한다. |
| 재현 기준 | 방향·범위 재현. 성공을 `final train accuracy>=.8`로 정의. BN on의 성공 scale 개수가 no-BN보다 크고, 성공 log-scale 범위가 최소 2배 넓어야 한다. BN on은 16개 중 12개 이상 성공을 운영 목표로 한다. |
| 분석 시행 세트 | `BN-OFF-01..16`, `BN-ON-01..16` |

### e06. Weight decay에 의한 overfitting 억제 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | MNIST train 앞 300개와 공식 test. 의도적으로 과적합되는 저데이터 분류. |
| 실험 목적 | L2 weight decay 강도가 weight norm, train-test gap 및 test 성능에 미치는 영향을 재현한다. |
| 사전 가설 | 무정규화 조건은 train accuracy가 거의 1에 도달하며 큰 generalization gap을 만든다. 중간 λ는 gap을 줄이지만 과도한 λ는 underfitting을 유발한다. |
| 독립변인 | weight decay `λ={0,1e-4,1e-3,1e-2,1e-1}`. |
| 고정변수 | MLP `784-[100×6]-10`, ReLU, He, SGD `lr=.01`, batch 100. 공통 301-epoch run 중 e06 분석은 epoch 0–200만 사용. dropout·BN off. |
| 종속변인 | train/test accuracy, generalization gap, best/final test accuracy, weight norm, loss. |
| 목적 확인을 위한 분석 | λ에 따른 학습 곡선과 최종 gap의 dose-response를 분석한다. test 결과로 최적 λ를 사후 선택하지 않고 sweep 전체를 보고한다. |
| 재현 기준 | 패턴 재현. `λ=0`의 median train accuracy `>=.98`. 양수 λ 중 적어도 하나가 baseline보다 median gap을 20% 이상 줄이고 test accuracy를 2%p 초과해 악화시키지 않아야 한다. |
| 분석 시행 세트 | `REG-BASE`, `REG-WD-1E4`, `REG-WD-1E3`, `REG-WD-1E2`, `REG-WD-1E1` |

### e07. Dropout에 의한 overfitting 억제 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | MNIST train 앞 300개와 공식 test. 저데이터 분류. |
| 실험 목적 | Dropout ratio에 따른 과적합 억제와 underfitting 경계를 재현한다. |
| 사전 가설 | dropout 0은 과적합하고, 중간 ratio는 gap을 줄이며, 큰 ratio는 학습 속도와 train 성능을 저하시킨다. |
| 독립변인 | dropout ratio `{0,.1,.2,.3,.5}`. |
| 고정변수 | MLP `784-[100×6]-10`, ReLU, He, SGD `lr=.01`, batch 100, 301 epochs, weight decay 0, BN off. |
| 종속변인 | train/test accuracy, generalization gap, best/final test accuracy, 수렴 속도. |
| 목적 확인을 위한 분석 | ratio별 gap과 test accuracy의 dose-response를 보고하고 `.5`에서 underfitting 여부를 확인한다. |
| 재현 기준 | 패턴 재현. ratio 0의 median train accuracy `>=.98`. `.1–.3` 중 하나가 baseline gap을 20% 이상 줄이며 test accuracy를 유지 또는 개선. `.5`는 중간 ratio보다 train accuracy 또는 수렴 속도가 낮아지는 패턴을 허용한다. |
| 분석 시행 세트 | `REG-BASE`, `REG-DO-01`, `REG-DO-02`, `REG-DO-03`, `REG-DO-05` |

### e08. CNN 구조 깊이에 따른 MNIST 성능 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | MNIST `(1,28,28)`. 10-class 이미지 분류. |
| 실험 목적 | 책의 SimpleConvNet과 DeepConvNet 전체 recipe가 정확도와 계산비용에서 보이는 차이를 재현한다. |
| 사전 가설 | DeepConvNet은 더 큰 계산비용을 사용하지만 SimpleConvNet보다 높은 정확도를 달성한다. |
| 독립변인 | architecture recipe `{SimpleConvNet,DeepConvNet}`. |
| 고정변수 | MNIST split, Adam `lr=.001`, batch 100, 20 epochs, 평가 방식. 단, 모델별 초기화·dropout·채널 구성은 recipe 일부로서 다르다. |
| 종속변인 | train/test accuracy, loss, parameter count, MACs/FLOPs, epoch time, inference throughput, peak memory. |
| 목적 확인을 위한 분석 | 정확도뿐 아니라 비용-성능 trade-off를 함께 비교한다. 이 결과를 순수한 depth 인과효과가 아니라 architecture recipe 비교로 해석한다. |
| 재현 기준 | 수치·방향 재현. DeepConvNet 10-seed median test accuracy `>=99.0%`, SimpleConvNet 운영 목표 `>=98.5%`. Deep이 paired mean 기준 Simple보다 `>=.2%p` 높아야 한다. |
| 분석 시행 세트 | `CNN-SIMPLE`, `CNN-DEEP` |

### e09. 추론 정밀도 float64/float32/float16 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | e08 DeepConvNet의 seed별 checkpoint와 MNIST test 10,000. 추론 평가. |
| 실험 목적 | 동일한 학습 결과를 낮은 dtype으로 변환했을 때 예측 및 정확도가 보존되는지, 성능 이점이 있는지 평가한다. |
| 사전 가설 | float32와 float16은 float64 기준 예측을 거의 보존하지만, 속도 이점은 backend와 hardware에 따라 달라진다. |
| 독립변인 | inference dtype `{float64,float32,float16}`. |
| 고정변수 | 동일 source checkpoint·입력·batch 100·evaluation mode. latency는 warm-up 후 동일 반복수로 측정. |
| 종속변인 | accuracy, float64 대비 prediction agreement, logit MAE/max error, latency, throughput, peak memory. |
| 목적 확인을 위한 분석 | 각 source seed 안에서 dtype을 paired 비교한다. 정확도와 prediction agreement를 주 분석으로 두고 속도는 backend별로 분리 보고한다. |
| 재현 기준 | 비열등 재현. float32·float16 accuracy drop `<=.1%p`; float16 prediction agreement `>=99.9%`. 속도 향상은 필수 기준으로 사용하지 않는다. |
| 분석 시행 세트 | `DTYPE-F64`, `DTYPE-F32`, `DTYPE-F16` |

### e10. CBOW vs Skip-gram 단어 임베딩 학습 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | PTB train. self-supervised 문맥-단어 예측 및 단어 임베딩 학습. |
| 실험 목적 | 동일 negative-sampling objective에서 CBOW와 Skip-gram의 학습 비용·loss·정성적 임베딩 품질을 비교한다. |
| 사전 가설 | 두 모델 모두 loss가 감소한다. Skip-gram은 target당 예측 항이 많아 계산비용이 크며, 품질 특성은 query에 따라 달라질 수 있다. |
| 독립변인 | architecture `{CBOW,SkipGram}`. |
| 고정변수 | PTB corpus/vocab, window 5, embedding 100, negative samples 5, noise power .75, batch 100, 10 epochs, Adam `.001`. |
| 종속변인 | 문맥 단어 1개당 normalized loss, throughput, epoch time, memory, nearest-neighbor와 analogy 결과. |
| 목적 확인을 위한 분석 | raw loss를 직접 비교하지 않고 prediction term당 normalized loss를 사용한다. 정성 평가 query는 고정하고 seed별 top-k 안정성을 분석한다. |
| 재현 기준 | 패턴 재현. 두 모델 모두 first-to-final epoch normalized loss 50% 이상 감소, NaN 0/10. Skip-gram의 epoch time 또는 연산량이 CBOW보다 커야 한다. |
| 분석 시행 세트 | `W2V-CBOW-NS`, `W2V-SG-NS` |

### e11. Naive CBOW full softmax와 negative sampling 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | PTB train. CBOW 중심어 예측. |
| 실험 목적 | 모델·데이터를 동일하게 유지하고 출력 objective만 바꿨을 때 full softmax와 negative sampling의 계산·메모리·품질 trade-off를 평가한다. |
| 사전 가설 | negative sampling은 full softmax보다 처리량과 메모리 사용에서 유리하며, 임베딩 품질은 크게 열화되지 않는다. |
| 독립변인 | objective `{full vocabulary softmax, negative sampling(sample=5,power=.75)}`. |
| 고정변수 | integer-index Embedding CBOW, PTB, window 5, embedding 100, batch 100, 10 epochs, Adam `.001`, 동일 초기 input embedding. |
| 종속변인 | normalized loss, throughput, peak memory, output-layer 연산량, nearest-neighbor·analogy 품질. |
| 목적 확인을 위한 분석 | 시스템 비용을 주 분석으로, 임베딩 품질을 비열등성 보조 분석으로 둔다. |
| 재현 기준 | 방향 재현. negative sampling이 full softmax보다 throughput이 높고 peak memory가 낮아야 한다. 고정 analogy set의 top-5 hit 수는 full softmax보다 1개를 초과해 감소하지 않는 것을 운영 목표로 한다. |
| 분석 시행 세트 | `W2V-CBOW-FULL`, `W2V-CBOW-NS` |

### e12. Simple RNN vs LSTM 언어모델 perplexity 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | PTB train/valid/test. next-token language modeling. |
| 실험 목적 | 동일한 모델 크기와 학습법에서 recurrent cell만 변경해 장기 의존성 처리와 perplexity 차이를 비교한다. |
| 사전 가설 | LSTM은 vanilla RNN보다 낮은 validation/test perplexity와 높은 학습 안정성을 보인다. |
| 독립변인 | recurrent cell `{vanilla RNN,LSTM}`. |
| 고정변수 | vocab 10,000, embedding/hidden 100, recurrent layer 1, batch 20, BPTT 35, SGD `lr=20`, max_grad `.25`, 4 epochs. |
| 종속변인 | train/valid/test perplexity, gradient norm, divergence rate, epoch time, parameter count. |
| 목적 확인을 위한 분석 | seed별 final test PPL과 curve AUC를 paired 비교한다. 실패 run은 제거하지 않고 실패율을 별도 종속변수로 보고한다. |
| 재현 기준 | 방향 재현. LSTM median test PPL이 RNN보다 15% 이상 낮고 paired difference 95% CI가 0 아래. LSTM 발산률은 RNN 이하. |
| 분석 시행 세트 | `LM-RNN-C025`, `LM-LSTM-C025` |

### e13. Gradient clipping threshold 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | PTB 1-layer LSTM language model. |
| 실험 목적 | gradient clipping threshold가 폭주 억제, update scale, 학습속도와 최종 perplexity에 미치는 영향을 평가한다. |
| 사전 가설 | clipping 없음은 큰 gradient norm과 불안정성을 보일 수 있고, 지나치게 작은 threshold는 학습을 늦춘다. 공식값 `.25`는 안정성과 학습성의 균형점이다. |
| 독립변인 | `max_grad={None,.1,.25,.5,1.0,5.0}`. |
| 고정변수 | 기본 LSTM RNNLM, PTB, embedding/hidden 100, batch 20, BPTT 35, SGD `lr=20`, 4 epochs. |
| 종속변인 | clip 전 gradient norm, clipping 발생률, applied scale, train/test PPL, divergence rate. |
| 목적 확인을 위한 분석 | threshold별 stability-performance 곡선을 분석한다. `.25`를 사전 지정 기준점으로 두고 no-clip 및 인접 threshold와 paired 비교한다. |
| 재현 기준 | 패턴 재현. `.25`는 10/10 finite loss를 유지하고 no-clip보다 median test PPL이 나쁘지 않아야 한다. `.1`은 `.25`보다 clipping 비율이 높아야 하며, no-clip은 최대 gradient norm이 모든 clipping 조건보다 커야 한다. |
| 분석 시행 세트 | `LM-LSTM-NONE`, `LM-LSTM-C010`, `LM-LSTM-C025`, `LM-LSTM-C050`, `LM-LSTM-C100`, `LM-LSTM-C500` |

### e14. 기본 RNNLM vs BetterRnnlm 개선 효과 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | PTB train/valid/test. next-token language modeling. |
| 실험 목적 | 2-layer LSTM, dropout, weight tying, 큰 hidden dimension과 validation 기반 LR decay가 결합된 BetterRnnlm recipe의 종합 효과를 재현한다. |
| 사전 가설 | BetterRnnlm recipe는 계산비용과 파라미터 규모가 크지만 기본 RNNLM보다 validation/test perplexity가 낮다. |
| 독립변인 | 전체 recipe `{basic RNNLM,BetterRnnlm}`. |
| 고정변수 | PTB vocabulary/split와 perplexity 산출법. 나머지 모델 용량과 training policy 차이는 독립변인 recipe에 포함한다. |
| 종속변인 | best validation PPL, test PPL, params, total time, peak memory, LR schedule. |
| 목적 확인을 위한 분석 | 최고 validation checkpoint로 test를 1회 평가한다. 비용 대비 PPL 개선을 함께 보고하며 개별 개선 요소의 인과효과로 해석하지 않는다. |
| 재현 기준 | 운영 수치 재현. Basic test PPL `<=150`, Better test PPL `<=90`, Better가 Basic보다 25% 이상 낮아야 한다. |
| 분석 시행 세트 | `LM-LSTM-C025`, `LM-BETTER` |

### e15. Seq2seq 입력 반전 효과 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | `addition.txt` 고정 90/10 split. character-level 덧셈 답 생성. |
| 실험 목적 | 입력 순서 반전이 encoder-decoder의 정보 전달 거리와 exact-match 학습을 개선하는지 확인한다. |
| 사전 가설 | reverse 입력은 vanilla Seq2seq의 수렴 속도와 최종 exact-match accuracy를 높인다. |
| 독립변인 | input reverse `{false,true}`. |
| 고정변수 | Vanilla Seq2seq, embedding 16, hidden 128, batch 128, 25 epochs, Adam `.001`, clip 5, greedy decoding. |
| 종속변인 | epoch별 sequence exact-match, token accuracy, loss, accuracy AUC, 목표 accuracy 도달 epoch. |
| 목적 확인을 위한 분석 | 같은 seed의 forward/reverse를 paired 비교한다. 최종 정확도와 전체 학습 곡선 AUC를 모두 사용한다. |
| 재현 기준 | 방향 재현. reverse final median exact-match `>=40%`, forward보다 `>=20%p` 높고 8/10 seed 이상에서 reverse가 우세. |
| 분석 시행 세트 | `SEQA-VAN-FWD`, `SEQA-VAN-REV` |

### e16. Seq2seq vs Peeky Seq2seq 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | `addition.txt`. character-level 덧셈 답 생성. |
| 실험 목적 | encoder hidden state를 decoder의 모든 step과 출력층에 공급하는 Peeky 구조가 fixed context 활용을 개선하는지 확인한다. |
| 사전 가설 | Peeky Seq2seq는 reverse vanilla Seq2seq보다 빠르고 높은 exact-match accuracy를 달성한다. |
| 독립변인 | architecture `{Vanilla Seq2seq,Peeky Seq2seq}`. |
| 고정변수 | reverse=true, embedding 16, hidden 128, batch 128, 25 epochs, Adam `.001`, clip 5, 동일 split·decode. |
| 종속변인 | sequence exact-match, token accuracy, loss, accuracy AUC, 목표 accuracy 도달 epoch, 연산비용. |
| 목적 확인을 위한 분석 | 동일 seed paired 비교. 수렴 속도와 최종 성능을 함께 평가한다. |
| 재현 기준 | 방향·운영 수치 재현. Peeky final median exact-match `>=90%`, Vanilla+reverse보다 `>=20%p` 높고 accuracy AUC도 높아야 한다. |
| 분석 시행 세트 | `SEQA-VAN-REV`, `SEQA-PEEKY-REV` |

### e17. Attention Seq2seq 성능 비교

| 항목 | 정의 |
| --- | --- |
| 데이터·태스크 | `date.txt` 고정 90/10 split. character-level 날짜 형식 변환. |
| 실험 목적 | Attention이 fixed-length encoder representation의 병목을 완화하고 입력-출력 alignment를 학습하는지 확인한다. |
| 사전 가설 | Attention Seq2seq는 Vanilla·Peeky보다 빠르게 높은 exact-match accuracy에 도달하며 해석 가능한 alignment를 생성한다. |
| 독립변인 | architecture `{Vanilla,Peeky,Attention Seq2seq}`. |
| 고정변수 | reverse=true, embedding 16, hidden 256, batch 128, 10 epochs, Adam `.001`, clip 5, greedy decoding. |
| 종속변인 | sequence exact-match, token accuracy, accuracy AUC, 목표 accuracy 도달 epoch, params/time, attention weight와 entropy. |
| 목적 확인을 위한 분석 | 세 architecture의 paired curve를 비교하고 Attention의 대표 alignment map을 정성·정량 분석한다. |
| 재현 기준 | 방향·운영 수치 재현. Attention final median exact-match `>=90%`, 세 조건 중 accuracy AUC 1위. 대표 샘플에서 출력 문자와 관련 입력 위치 사이의 alignment가 관찰되어야 한다. |
| 분석 시행 세트 | `SEQD-VAN-REV`, `SEQD-PEEKY-REV`, `SEQD-ATTN-REV` |

## 5. 해석 제한

- e08은 depth만 통제한 실험이 아니라 SimpleConvNet과 DeepConvNet의 전체 architecture recipe 비교다.
- e14는 BetterRnnlm 구성요소 각각의 ablation이 아니라 전체 model/training recipe 비교다.
- e09의 속도 결과는 dtype 자체뿐 아니라 NumPy/CuPy, CPU/GPU, kernel 지원과 hardware에 의존한다.
- e10의 CBOW와 Skip-gram raw loss는 예측 항의 개수가 달라 직접 비교하지 않는다.
- e06·e07의 공식 예제는 test curve를 학습 중 반복 관찰한다. 본 프로젝트에서는 sweep 전체를 보고하며 test를 이용한 사후 최적 조건 선택을 하지 않는다.

## 6. 공식 코드 기준 파일

- `oreilly-japan/deep-learning-from-scratch/ch06/optimizer_compare_naive.py`
- `oreilly-japan/deep-learning-from-scratch/ch06/optimizer_compare_mnist.py`
- `oreilly-japan/deep-learning-from-scratch/ch06/weight_init_activation_histogram.py`
- `oreilly-japan/deep-learning-from-scratch/ch06/weight_init_compare.py`
- `oreilly-japan/deep-learning-from-scratch/ch06/batch_norm_test.py`
- `oreilly-japan/deep-learning-from-scratch/ch06/overfit_weight_decay.py`
- `oreilly-japan/deep-learning-from-scratch/ch06/overfit_dropout.py`
- `oreilly-japan/deep-learning-from-scratch/ch07/train_convnet.py`
- `oreilly-japan/deep-learning-from-scratch/ch08/train_deepnet.py`
- `oreilly-japan/deep-learning-from-scratch/ch08/half_float_network.py`
- `oreilly-japan/deep-learning-from-scratch-2/ch04/train.py`
- `oreilly-japan/deep-learning-from-scratch-2/ch05/train.py`
- `oreilly-japan/deep-learning-from-scratch-2/ch06/train_rnnlm.py`
- `oreilly-japan/deep-learning-from-scratch-2/ch06/train_better_rnnlm.py`
- `oreilly-japan/deep-learning-from-scratch-2/ch07/train_seq2seq.py`
- `oreilly-japan/deep-learning-from-scratch-2/ch08/train.py`
