# 실행 그룹 구현 명세

이 문서는 구현 담당자용 실행 명세다. 원본 출처·원본 시행 ID·책의 관측 방식은 여기서 다루지 않는다. 학습 그룹은 `GTnn`, 관찰 그룹은 `GOnn`으로 식별한다.

기록 파일·열·MLflow metric 이름과 그룹별 evaluation schedule은 [`recording_schema.md`](recording_schema.md)에 고정한다.

## 그룹화 계약

실행 그룹은 하나의 완결된 실행 protocol이다. 같은 그룹의 원자 시행은 dataset/split/subset, 학습 budget, sampler, optimizer update 단위, 평가 cadence, checkpoint 정책, seed-pairing 정책을 공유한다.

원자 시행 간 차이는 해당 그룹의 `변동 축`에 선언한 필드로 제한한다. 이 규칙은 모든 그룹에 동일하게 적용한다. 변동 축이 없는 singleton 그룹도 하나의 canonical protocol로 유효하다.

모든 확률적 원자 시행의 실제 run ID는 `<atomic_run_id>.seed-<n>`이다. 같은 그룹 내 paired 조건은 동일 master seed를 사용한다.

## 관찰 그룹

| 그룹   | protocol                          | 공통 조건                                                       | 변동 축                          | 원자 시행 수 |
| ------ | --------------------------------- | --------------------------------------------------------------- | -------------------------------- | -----------: |
| `GO01` | analytic optimization observation | `f=x²/20+y²`, init `(-7,2)`, 30 updates, float64                | optimizer + optimizer parameters |            4 |
| `GO02` | activation forward observation    | synthetic normal input `(1000,100)`, width 100, depth 5, bias 0 | activation × initializer         |           12 |

## 학습 그룹

| 그룹   | protocol                        | 공통 조건                                                                                      | 변동 축                               | 원자 시행 수 |
| ------ | ------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------- | -----------: |
| `GT01` | MNIST MLP optimization          | MNIST full train, MLP `784-[100×4]-10`, batch 128, 2,000 updates                               | optimizer + optimizer parameters      |            4 |
| `GT02` | MNIST MLP initialization        | MNIST full train, MLP `784-[100×4]-10`, SGD `.01`, batch 128, 2,000 updates                    | initializer                           |            3 |
| `GT03` | MNIST MLP weight decay          | MNIST train first 300 + official test, MLP `784-[100×6]-10`, SGD `.01`, batch 100, 201 epochs  | weight-decay lambda                   |            2 |
| `GT04` | MNIST MLP dropout               | MNIST train first 300 + official test, MLP `784-[100×6]-10`, SGD `.01`, batch 100, 301 epochs  | dropout flag + ratio                  |            2 |
| `GT05` | MNIST MLP BatchNorm scale       | MNIST train first 1,000, MLP `784-[100×5]-10`, SGD `.01`, batch 100, 20 epochs                 | BatchNorm flag × initialization scale |           32 |
| `GT06` | MNIST SimpleConvNet             | MNIST image full train/test, SimpleConvNet, Adam `.001`, batch 100, 20 epochs                  | 없음                                  |            1 |
| `GT07` | MNIST DeepConvNet               | MNIST image full train/test, DeepConvNet, Adam `.001`, batch 100, 20 epochs                    | 없음                                  |            1 |
| `GT08` | MNIST spatial-layout comparison | MNIST full train/test, Adam `.001`, batch 100, replacement sampling, 2 epochs, 10 paired seeds | architecture × input transform        |            4 |
| `GT09` | MNIST extended MLP comparison   | MNIST full train/test, MLP `784-[100×6]-10`, Adam `.001`, batch 100, 20 epochs               | 없음                                  |            1 |

## 현재 executor 재현 상태

이 표는 문서의 **목표 protocol**이 아니라, 2026-07-22 현재
`exp/deepscratch/ds1/implemented/executor.py`와 `exp/deepscratch/ds1/config/implemented/`으로 실제 실행할 때의 상태다. `부분`
그룹의 결과는 해당 원본 그래프의 완전 재현으로 사용하면 안 된다.

| 그룹 | 상태 | 현재 가능한 범위 | 재현되지 않는 사항 |
| --- | --- | --- | --- |
| `GO01` | 미구현 | 없음 | active catalog/executor가 없다. analytic trajectory artifact와 4 optimizer 조건 모두 없다. |
| `GO02` | 미구현 | 없음 | active catalog/executor가 없다. activation histogram/summary artifact와 12 조건 모두 없다. |
| `GT01` | 완료 | 4 optimizer, 2,000 post-update loss, paired seed | schedule상 중간 accuracy는 요구하지 않는다. |
| `GT02` | 완료 | 3 initializer, 2,000 post-update loss, paired seed | schedule상 중간 accuracy는 요구하지 않는다. |
| `GT03` | 완료 | 2 weight-decay 조건, 601 updates, `1,4,7,...` cadence, train-first-300/test-full 평가 | 없음. |
| `GT04` | 완료 | 2 dropout 조건, epoch 첫 update에서 train-first-300/test-full 평가 | 없음. |
| `GT05` | 완료 | 32 BatchNorm×scale 조건, 20 epochs, `1,11,21,...,191` cadence, train-first-1000 평가 | 없음. |
| `GT06` | 완료 | SimpleCNN, 20 epochs, epoch-first train-first-1000/test-first-1000, terminal test-full 평가 | 없음. |
| `GT07` | 완료 | DeepCNN, 20 epochs, epoch-first train-first-1000/test-first-1000, terminal test-full 평가 | 없음. |
| `GT08` | 완료 | 4 spatial/input-transform 조건, paired seed, 20-update train/test first-1000 평가, epoch-end test-full 평가 | 없음. |
| `GT09` | 구현 완료 | ReLU, He, BatchNorm, dropout `.2`, weight decay `.1`을 적용한 6-hidden-layer MLP; GT07과 같은 학습·평가 protocol | 실행 결과는 아직 생성하지 않았다. |

### 공통 구현 제한

- `updates.csv`/`evaluations.csv`/`timing_windows.csv`는 256-record, epoch 종료,
  checkpoint 직전, run 종료에 flush한다.
- `checkpoints.csv`는 schema header와 checkpoint hash를 기록한다.
- GPU profile-mode device timing은 CUDA event 기반 window timing으로 기록한다.

완전 재현을 위한 남은 우선순위는 문서 스키마와 artifact 검증을 계속 맞추는 것이다.

## GO01 — optimizer trajectory observation

| atomic run ID  | optimizer                        |
| -------------- | -------------------------------- |
| `TOY-SGD`      | SGD, `lr=.95`                    |
| `TOY-MOMENTUM` | Momentum, `lr=.1`, `momentum=.9` |
| `TOY-ADAGRAD`  | AdaGrad, `lr=1.5`                |
| `TOY-ADAM`     | Adam, `lr=.3`                    |

## GT01 — MNIST optimizer

| atomic run ID      | optimizer                         |
| ------------------ | --------------------------------- |
| `MLP-OPT-SGD`      | SGD, `lr=.01`                     |
| `MLP-OPT-MOMENTUM` | Momentum, `lr=.01`, `momentum=.9` |
| `MLP-OPT-ADAGRAD`  | AdaGrad, `lr=.01`                 |
| `MLP-OPT-ADAM`     | Adam, `lr=.001`                   |

## GO02 — activation/initialization observation

입력은 같은 seed에서 모든 조건이 공유한다.

| atomic run ID pattern        | activation          | initializer      |
| ---------------------------- | ------------------- | ---------------- |
| `ACT-{SIG,TANH,RELU}-STD1`   | sigmoid, tanh, ReLU | Normal std `1`   |
| `ACT-{SIG,TANH,RELU}-STD001` | sigmoid, tanh, ReLU | Normal std `.01` |
| `ACT-{SIG,TANH,RELU}-XAVIER` | sigmoid, tanh, ReLU | `sqrt(1/fan_in)` |
| `ACT-{SIG,TANH,RELU}-HE`     | sigmoid, tanh, ReLU | `sqrt(2/fan_in)` |

## GT02 — initialization comparison

| atomic run ID     | initializer             |
| ----------------- | ----------------------- |
| `MLP-INIT-STD001` | Normal std `.01`        |
| `MLP-INIT-XAVIER` | Xavier `sqrt(1/fan_in)` |
| `MLP-INIT-HE`     | He `sqrt(2/fan_in)`     |

## GT03 — weight decay

| atomic run ID | `weight_decay_lambda` |
| ------------- | --------------------: |
| `REG-WD-OFF`  |                   `0` |
| `REG-WD-01`   |                  `.1` |

## GT04 — dropout

| atomic run ID       | `use_dropout` | `dropout_ratio` |
| ------------------- | ------------- | --------------: |
| `REG-DROPOUT-OFF`   | false         |             `0` |
| `REG-DROPOUT-ON-02` | true          |            `.2` |

## GT05 — BatchNorm × initialization scale

`k=01..16`, `scale-k = logspace(0,-4,16)[k-1]`로 정의한다.

| atomic run ID pattern | `use_batchnorm` | `weight_init_std` |
| --------------------- | --------------- | ----------------- |
| `BN-SCALE-{k}-OFF`    | false           | `scale-k`         |
| `BN-SCALE-{k}-ON`     | true            | `scale-k`         |

## GT06 — SimpleConvNet

| atomic run ID     | model                                                          |
| ----------------- | -------------------------------------------------------------- |
| `CNN-SIMPLE-BOOK` | Conv30 5×5 → ReLU → Pool → FC100 → ReLU → FC10, init std `.01` |

## GT07 — DeepConvNet

| atomic run ID   | model                                                            |
| --------------- | ---------------------------------------------------------------- |
| `CNN-DEEP-BOOK` | 6-convolution DeepConvNet, FC50, dropout `.5`, He initialization |

## GT08 — spatial-layout comparison

같은 master seed의 원본/순열 조건은 paired run이다. `pixel_permutation_seed=20260808`로 고정한 permutation을 train/test에 함께 적용한다.

| atomic run ID                 | architecture                                        | input transform         |
| ----------------------------- | --------------------------------------------------- | ----------------------- |
| `NN-MATCHED`                  | ParameterMatchedNN `784-489-100-10`, ReLU, He       | identity                |
| `NN-MATCHED-PERMUTED`         | ParameterMatchedNN `784-489-100-10`, ReLU, He       | fixed pixel permutation |
| `CNN-SIMPLE-SPATIAL`          | SimpleConvNet `Conv30 5×5 → FC100`, ReLU, std `.01` | identity                |
| `CNN-SIMPLE-SPATIAL-PERMUTED` | SimpleConvNet `Conv30 5×5 → FC100`, ReLU, std `.01` | fixed pixel permutation |

각 trial manifest에는 architecture signature, parameter count, master seed, `pixel_permutation_seed`, permutation checksum을 기록한다.

## GT09 — extended MLP versus DeepConvNet

| atomic run ID       | model                                                                 |
| ------------------- | --------------------------------------------------------------------- |
| `MLP-EXT-ALL-BOOK`  | MLP `784-[100×6]-10`, ReLU, He, BatchNorm, dropout `.2`, L2 decay `.1` |

GT09는 GT07의 기존 `CNN-DEEP-BOOK` run을 다시 실행하지 않는다. 데이터 크기, batch
sampling, Adam `.001`, 20-epoch budget, seed set과 evaluation cadence를 GT07과 동일하게
두고, `e12` 분석에서 두 실행 그룹을 함께 읽어 비교한다. MLP 입력만 모델 요구사항에
따라 `(N, 784)`로 flatten한다.

## 원본 시행 ID

| 구현 그룹 ID | 원본 시행 ID |
| --- | --- |
| `GT01` | `SRC-B1-CH06-OPTIMIZER-MNIST` |
| `GT02` | `SRC-B1-CH06-WEIGHT-INIT-COMPARE` |
| `GT03` | `SRC-B1-CH06-WEIGHT-DECAY` |
| `GT04` | `SRC-B1-CH06-DROPOUT` |
| `GT05` | `SRC-B1-CH06-BATCHNORM-SCALE` |
| `GT06` | `SRC-B1-CH07-SIMPLE-CONVNET` |
| `GT07` | `SRC-B1-CH08-DEEP-CONVNET` |
| `GO01` | `SRC-B1-CH06-OPTIMIZER-PATH` |
| `GO02` | `SRC-B1-CH06-ACTIVATION-HISTOGRAM` |

`GT08`과 `GT09`는 원본 시행 ID를 갖지 않는 새 확장 시행이다.
