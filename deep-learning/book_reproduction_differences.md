# 책 원본 대비 실험 재현 차이

이 문서는 현재 저장된 DS1·DS2 재현 결과를 책 원본 코드 실행 결과와 비교해, 실험별로 무엇이 얼마나 다른지 정리한다.

## 비교 기준

- 원본 값은 저장된 책 코드 실행 결과, 책 코드 체크포인트 평가값, 또는 실행 노트북 출력값이다.
- 현재 값은 재현 실험의 seed 집계다. 10회 시행 결과는 모두 `평균 ± 표본 표준편차`로 표기한다.
- 그래프의 10회 시행 오차 음영도 각 지점의 `평균 ± 표본 표준편차` 범위다.
- `차이 = 현재 값 - 원본 값`이다. 정확도 차이는 percentage point(`pp`)로 적는다.
- 원본은 대부분 단일 실행이고 현재 값은 다중 seed 평균이므로, 차이가 구현 오차만을 의미하지는 않는다.
- `원본 미캐시`는 책에 조건은 있지만 완료된 원본 실행 결과가 저장소에 없다는 뜻이다.
- `직접 대응 없음`은 현재 실험이 책 구성요소를 이용한 프로젝트 확장이라는 뜻이다.

근거 문서와 결과:

- [DS1 원본 시행 레지스트리](exp/ds1/docs/trial_registry.md)
- [DS2 원본 시행 레지스트리](exp/ds2/docs/trial_registry.md)
- [DS1 재현 설정](exp/ds1/config)
- [DS2 재현 설정](exp/ds2/config)
- [DS1 원본 결과](exp/ds1/results/original)
- [DS2 원본 결과](exp/ds2/results/original)
- [PPT 생성 스크립트](build_ds_presentation.py)

## 차이 요약

| 실험 | 설정 차이 | 결과 차이 요약 |
| --- | --- | --- |
| MNIST Optimizer Comparison | 핵심 설정 일치, 단일 실행 → 10 paired seeds | 최종 loss 차이 `-0.007`~`+0.037` |
| MNIST Weight Initialization | 핵심 설정 일치, 단일 실행 → 10 paired seeds | 최종 loss `0.005`~`0.060` 낮음 |
| Weight Decay and Overfitting | no-decay 대조군 추가 | weight decay 조건 test accuracy `+1.70 pp` |
| Dropout and Overfitting | dropout-off 대조군 추가 | dropout 0.2 조건 test accuracy `-2.28 pp` |
| Batch Normalization and Weight Scale | 핵심 설정 일치, 각 조건 10회 반복 | 대부분 근접하나 일부 작은 weight scale에서 최대 `+17.80 pp` |
| Simple Convolutional Network | 다중 seed와 GPU/numerics backend 적용 | full-test accuracy `-0.05 pp` |
| Deep Convolutional Network | 다중 seed와 GPU/numerics backend 적용 | full-test accuracy `-0.41 pp` |
| Spatial Layout Sensitivity | 책에 없는 확장 실험 | 직접 수치 비교 불가 |
| Optimizer Trajectories | 30 → 60 updates | 종료 시점이 달라 최종 좌표 직접 비교 불가 |
| Activation Distribution Observation | 1조건 → 12조건, seed 정책 변경 | 공통 조건 histogram peak `1.5%`~`24.4%` 낮음 |
| First-Layer Convolution Filters | initial/trained 비교 → 3개 trained 조건 비교 | 공통 SimpleCNN filter std `-0.006` |
| Toy Word2Vec | Skip-gram 및 10-seed 비교 추가 | CBOW final loss `-0.247` |
| PTB Word2Vec | full-softmax 2조건 추가 | CBOW negative-sampling final loss 약 `-0.020` |
| Small-Corpus RNN Language Model | 단일 실행 → 10 deterministic GPU seeds | train perplexity `-0.22` |
| Penn Treebank LSTM Language Model | 단일 checkpoint → 10 deterministic GPU seeds | test perplexity `-1.29` |
| Penn Treebank LM Recipes | 책 모델을 하나의 비교 실험으로 통합 | improved LSTM test perplexity `-1.81` |
| Addition Seq2Seq | 책의 개별 조건을 paired 비교로 통합 | vanilla/forward accuracy `-3.31 pp` |
| Date Conversion Seq2Seq | 책의 개별 조건을 paired 비교로 통합 | attention accuracy `-0.08 pp` |
| Attention Alignment | 프로토콜 일치, 현재 checkpoint 의존성 미충족 | 원본 5개 map, 현재 0개 |

## DS1

### MNIST Optimizer Comparison

- 설정:
  - 데이터, `784 → 100×4 → 10` MLP, batch `128`, optimizer별 learning rate, `2,000` updates는 원본과 같다.
  - 원본 단일 실행을 현재는 10개 paired seed로 반복해 평균·범위를 계산한다.
- 결과:

| Optimizer | 원본 final loss | 현재 평균 ± 표준편차 | 차이 |
| --- | ---: | ---: | ---: |
| SGD | 0.185 | 0.199 ± 0.021 | +0.014 |
| Momentum | 0.026 | 0.063 ± 0.015 | +0.037 |
| AdaGrad | 0.017 | 0.028 ± 0.005 | +0.011 |
| Adam | 0.046 | 0.039 ± 0.009 | -0.007 |

- 가장 큰 절대 차이는 Momentum의 `+0.037`이다. optimizer 순위와 전반적인 수렴 형태는 유지된다.

### MNIST Weight Initialization

- 설정:
  - MLP 구조, ReLU, SGD `0.01`, batch `128`, `2,000` updates, 세 초기화 조건은 원본과 같다.
  - 원본 단일 실행을 10개 paired seed 평균으로 확장했다.
- 결과:

| Initializer | 원본 final loss | 현재 평균 ± 표준편차 | 차이 |
| --- | ---: | ---: | ---: |
| Normal σ=0.01 | 2.306 | 2.301 ± 0.002 | -0.005 |
| Xavier | 0.324 | 0.264 ± 0.027 | -0.060 |
| He | 0.245 | 0.199 ± 0.021 | -0.046 |

- 초기화 조건의 상대적 결론은 동일하다. Xavier와 He는 현재 평균이 원본 단일 실행보다 각각 `0.060`, `0.046` 낮다.

### Weight Decay and Overfitting

- 설정:
  - 원본 source-default는 weight decay `0.1`이다.
  - 현재 재현은 동일 조건에 weight decay `0.0` 대조군과 10개 paired seed를 추가했다.
- 결과:
  - weight decay `0.1`: 원본 test accuracy `70.79%`, 현재 `72.48% ± 2.28%`, 차이 `+1.70 pp`.
  - no-decay 현재 값은 `75.68% ± 1.69%`지만 대응하는 원본 실행이 캐시되지 않아 직접 차이는 계산하지 않는다.

### Dropout and Overfitting

- 설정:
  - 원본 source-default는 dropout `0.2`이다.
  - 현재 재현은 dropout `0.0` 대조군과 10개 paired seed를 추가했다.
- 결과:
  - dropout `0.2`: 원본 test accuracy `55.66%`, 현재 `53.38% ± 7.83%`, 차이 `-2.28 pp`.
  - dropout-off 현재 값은 `76.08% ± 1.67%`지만 대응하는 원본 실행이 캐시되지 않았다.

### Batch Normalization and Weight Scale

- 설정:
  - `784 → 100×5 → 10` MLP, 16개 weight scale, BatchNorm on/off, SGD `0.01`, 20-epoch 관찰 cadence가 원본과 같다.
  - 현재는 32개 조건을 각각 10개 paired seed로 반복한다.
- 결과 차이:

| Initial σ | No BN 차이 | BN 차이 |
| ---: | ---: | ---: |
| 1.0 | -1.90 pp | -4.60 pp |
| 0.541170 | 0.00 pp | -1.37 pp |
| 0.292864 | +2.58 pp | -0.86 pp |
| 0.158489 | +0.17 pp | +1.41 pp |
| 0.085770 | -5.94 pp | -0.43 pp |
| 0.046416 | +0.50 pp | +0.01 pp |
| 0.025119 | +0.04 pp | -0.01 pp |
| 0.013594 | +0.04 pp | -0.14 pp |
| 0.007356 | +0.04 pp | -0.03 pp |
| 0.003981 | +0.04 pp | -0.25 pp |
| 0.002154 | +0.04 pp | -3.25 pp |
| 0.001166 | +0.04 pp | +17.80 pp |
| 0.000631 | +0.04 pp | +4.35 pp |
| 0.000341 | +0.04 pp | -0.51 pp |
| 0.000185 | +0.04 pp | -7.25 pp |
| 0.000100 | +0.04 pp | +9.42 pp |

- No-BN 조건은 대부분 근접하며 최대 절대 차이는 σ=`0.085770`의 `5.94 pp`다.
- BN 조건은 매우 작은 초기 weight scale에서 seed 민감도가 커지며 최대 차이는 σ=`0.001166`의 `+17.80 pp`다.

### Simple Convolutional Network

- 설정:
  - Conv30 `5×5 → FC100 → 10`, Adam `0.001`, batch `100`, 20 epochs는 원본과 같다.
  - 현재는 10개 seed 및 저장소의 CuPy/float64 backend로 실행한다.
- 결과:
  - 원본 full-test accuracy `98.82%`.
  - 현재 `98.77% ± 0.28%`.
  - 차이 `-0.05 pp`.

### Deep Convolutional Network

- 설정:
  - 6개 convolution layer, channel stages `16/32/64`, FC `50`, dropout `0.5`, Adam `0.001`, 20 epochs는 원본과 같다.
  - 현재는 10개 seed 및 저장소의 CuPy/float64 backend를 사용한다.
- 결과:
  - 원본 full-test accuracy `99.42%`.
  - 현재 `99.01% ± 0.13%`.
  - 차이 `-0.41 pp`.

### Spatial Layout Sensitivity

- 설정:
  - 책에 동일한 실험이 없다.
  - 책의 MLP와 SimpleCNN 구성요소에 고정 pixel permutation을 추가한 프로젝트 확장이다.
  - 원본/순열 입력을 parameter-matched MLP와 SimpleCNN에 각각 적용한다.
- 현재 결과:

| 조건 | Test accuracy (평균 ± 표준편차) |
| --- | ---: |
| MLP / original | 97.04% ± 0.46% |
| MLP / permuted | 97.29% ± 0.61% |
| CNN / original | 97.77% ± 0.73% |
| CNN / permuted | 93.83% ± 0.67% |

- MLP의 permutation 변화는 `+0.25 pp`, CNN은 `-3.94 pp`다. 책 원본과의 직접 차이는 계산할 수 없다.

### Optimizer Trajectories

- 설정:
  - 목적함수, 초기점, optimizer, learning rate는 원본과 같다.
  - 원본은 30개 pre-update 위치를 기록하고, 현재는 60 updates까지 확장한다.
- 결과:

| Optimizer | 원본 좌표 @30 | 현재 좌표 @60 |
| --- | --- | --- |
| SGD | (-0.387, -0.094) | (-0.019, -0.004) |
| Momentum | (0.814, 0.273) | (0.002, -0.048) |
| AdaGrad | (-0.136, 9.07e-17) | (-0.003, 1.24e-33) |
| Adam | (0.020, 0.322) | (0.138, 0.072) |

- 종료 update가 달라 좌표 차이를 구현 오차로 해석할 수 없다. 현재 결과는 원본보다 30 updates 더 진행된 상태다.

### Activation Distribution Observation

- 설정:
  - 원본 source-default는 sigmoid와 weight σ=`1.0`, NumPy seed `1` 한 조건이다.
  - 현재는 sigmoid/tanh/ReLU와 네 initializer의 `3×4` factorial로 확장하고 input/model seed를 분리했다.
- 공통 조건인 sigmoid/σ=`1.0` histogram peak 비교:

| Layer | 원본 peak count | 현재 peak count | 차이 | 상대 차이 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 36,690 | 34,777 | -1,913 | -5.2% |
| 2 | 31,654 | 26,294 | -5,360 | -16.9% |
| 3 | 35,421 | 34,902 | -519 | -1.5% |
| 4 | 38,862 | 30,818 | -8,044 | -20.7% |
| 5 | 38,833 | 29,372 | -9,461 | -24.4% |

- seed와 factorial protocol이 달라 histogram count를 직접 재현 오차로만 해석할 수 없다.

### First-Layer Convolution Filters

- 설정:
  - 원본은 동일 SimpleCNN의 initial filter와 trained filter를 비교한다.
  - 현재는 full, spatial, spatial-permuted 조건에서 학습된 filter 세트를 비교한다.
- 공통 trained SimpleCNN 첫 layer 통계:

| 값 | 원본 | 현재 | 차이 |
| --- | ---: | ---: | ---: |
| Minimum | -0.897 | -0.761 | +0.136 |
| Maximum | 0.575 | 0.447 | -0.128 |
| Standard deviation | 0.217 | 0.211 | -0.006 |

- tensor shape `30×1×5×5`는 동일하다.

## DS2

### Toy Word2Vec: CBOW vs Skip-gram

- 설정:
  - 책 원본은 toy CBOW만 실행한다.
  - 현재는 동일 CBOW에 toy Skip-gram을 추가하고 두 모델을 10개 paired seed로 반복한다.
- 결과:
  - CBOW 원본 final loss `0.878`, 현재 `0.631 ± 0.216`, 차이 `-0.247`.
  - Skip-gram 현재 값은 `1.882 ± 0.294`이며 대응하는 책 원본 실행은 없다.
- 원본과 현재의 seed protocol이 달라 CBOW 차이는 단일-run 재현 오차와 seed 평균 효과가 섞여 있다.

### PTB Word2Vec Objectives

- 설정:
  - CBOW/Skip-gram negative sampling은 책에서 선택 가능한 조건이다.
  - 현재는 두 full-softmax 조건과 10개 paired deterministic seed를 추가했다.
- 결과:

| 조건 | 원본 | 현재 평균 ± 표준편차 | 차이/상태 |
| --- | ---: | ---: | --- |
| CBOW / negative sampling | 약 1.49 | 1.470 ± 0.019 | 약 -0.020 |
| Skip-gram / negative sampling | 원본 미캐시 | 22.705 ± 0.113 | 책 조건, 비교 보류 |
| CBOW / full softmax | 직접 대응 없음 | 4.858 ± 0.068 | 프로젝트 확장 |
| Skip-gram / full softmax | 직접 대응 없음 | 62.058 ± 0.142 | 프로젝트 확장 |

### Small-Corpus RNN Language Model

- 설정:
  - PTB 첫 1,000 tokens, embedding/hidden `100`, batch `10`, BPTT `5`, SGD `0.1`, 100 epochs는 원본과 같다.
  - 현재는 단일 실행 대신 10개 deterministic GPU seed를 사용한다.
- 결과:
  - 원본 final train perplexity `6.10`.
  - 현재 `5.88 ± 0.61`.
  - 차이 `-0.22`, 원본 대비 약 `-3.6%`.

### Penn Treebank LSTM Language Model

- 설정:
  - embedding/hidden `100`, batch `20`, BPTT `35`, SGD `20`, gradient clipping `0.25`, 4 epochs는 원본과 같다.
  - 현재는 단일 checkpoint 평가 대신 10개 deterministic GPU seed 평균을 사용한다.
- 결과:
  - 원본 checkpoint test perplexity `136.08`.
  - 현재 `134.79 ± 0.92`.
  - 차이 `-1.29`, 원본 대비 약 `-0.95%`.

### Penn Treebank LM Recipes

- 설정:
  - 책의 Vanilla RNN, LSTM, improved LSTM 구조를 하나의 3-recipe 비교 실험으로 통합한 프로젝트 확장이다.
  - 책은 동일 형식의 통합 비교 실행을 제공하지 않는다.
- 결과:

| Recipe | 원본 | 현재 | 차이/상태 |
| --- | ---: | ---: | --- |
| Vanilla RNN | 통합 원본 없음 | 30,189.55 | 직접 비교 불가 |
| LSTM | 통합 원본 없음 | 117.76 | 직접 비교 불가 |
| Improved LSTM | 80.83 | 79.02 | -1.81 |

- improved LSTM의 현재 test perplexity는 원본보다 약 `2.2%` 낮다.

### Addition Seq2Seq Models

- 설정:
  - 책 source에서 개별 선택하는 Vanilla/Peeky와 forward/reversed 조건을 현재는 하나의 paired 실험으로 통합했다.
  - 각 조건은 10개 paired seed로 실행한다.
- 결과:

| 조건 | 원본 | 현재 평균 ± 표준편차 | 차이/상태 |
| --- | ---: | ---: | --- |
| Vanilla / forward | 12.94% | 9.63% ± 4.95% | -3.31 pp |
| Vanilla / reversed | 원본 미캐시 | 40.02% ± 12.16% | 비교 보류 |
| Peeky / forward | 원본 미캐시 | 28.49% ± 9.25% | 비교 보류 |
| Peeky / reversed | 원본 미캐시 | 89.40% ± 18.76% | 비교 보류 |

### Date Conversion Seq2Seq Models

- 설정:
  - 책 source에서 선택하는 Vanilla, Peeky, Attention 구조를 현재는 하나의 paired 실험으로 통합했다.
  - 세 조건 모두 10개 paired seed로 실행한다.
- 결과:

| 모델 | 원본 | 현재 평균 ± 표준편차 | 차이/상태 |
| --- | ---: | ---: | --- |
| Vanilla | 원본 미캐시 | 0.00% ± 0.01% | 비교 보류 |
| Peeky | 원본 미캐시 | 99.98% ± 0.05% | 비교 보류 |
| Attention | 100.00% | 99.92% ± 0.14% | -0.08 pp |

- 세 모델 모두 동일한 10개 seed 집합을 사용하며, 표준편차는 seed 간 변동을 나타낸다.

### Attention Alignment Observation

- 설정:
  - date test split, reversed input, selection seed `1984`, 5 examples, greedy decoding이라는 관찰 프로토콜은 원본과 같다.
  - 현재 관찰은 학습 완료된 AttentionSeq2seq checkpoint에 의존한다.
- 결과:

| 항목 | 원본 | 현재 |
| --- | ---: | ---: |
| 사용 가능한 trained checkpoint | 1 | 0 |
| Alignment examples | 5 | 0 |
| 생성된 attention map | 5 | 0 |

- 현재 차이는 수치 재현 오차가 아니라 checkpoint 의존성이 충족되지 않은 실행 상태 차이다.

## 해석 시 우선 확인할 항목

1. BatchNorm의 작은 weight scale 구간은 seed 분산이 커 원본 단일 실행과 평균 차이가 크게 나타난다.
2. Optimizer trajectory는 update budget이 다르므로 같은 종료 좌표를 기대하면 안 된다.
3. Activation observation은 공통 조건도 seed 정책이 다르며, 나머지 11조건은 프로젝트 확장이다.
4. DS2의 `원본 미캐시` 조건은 현재 결과가 잘못됐다는 뜻이 아니라 원본 수치 비교가 아직 불가능하다는 뜻이다.
5. 10회 시행 그래프의 음영은 min–max가 아니라 평균 ± 표본 표준편차이므로, 개별 실행의 전체 범위로 해석하면 안 된다.
6. Attention alignment는 현재 trained checkpoint를 확보한 뒤에야 실제 재현 차이를 계산할 수 있다.
