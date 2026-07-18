# 딥러닝 밑바닥 실험 재현: 원자 시행 및 실행 플랫폼 명세서

> 관점: ML engineering / MLOps 실행 플랫폼 설계
> 목표: 이 문서의 레시피와 원자 시행 registry만으로 실행 계획을 생성하고 MLflow에 중복 없이 기록한다.

## 1. 핵심 실행 모델

```text
Recipe + Atomic override = Parent condition
Parent condition + Child seed = 실제 학습/평가 1회
Analysis = 여러 Parent condition의 child 결과 조회
```

- atomic run ID 하나는 seed를 제외한 완전한 조건 하나다.
- 모든 확률적 atomic run은 child seed `0..9`를 요구한다.
- `RC-TOY` canonical은 결정적이므로 child를 만들지 않는다.
- `DTYPE-*`는 training이 아닌 derived evaluation이며 같은 seed의 `CNN-DEEP` checkpoint를 참조한다.
- Registry는 총 85개 atomic condition으로 구성된다: 결정적 toy 4개, seed 반복 training/probe 78개(780 children), derived dtype evaluation 3개(30 children).

## 2. 설정 해석 규칙

실행기는 다음 순서로 resolved config를 생성해야 한다.

```text
resolved = deep_merge(
    global_policy,
    dataset_registry[recipe.dataset],
    recipe_registry[atomic.recipe],
    atomic.override,
    child_seed_policy[seed],
)
```

필수 필드가 남아 있거나 알 수 없는 키가 존재하면 실행하지 않고 plan validation error로 종료한다. resolved config 전체를 정렬된 JSON으로 직렬화해 condition key와 child run key를 계산한다.

```text
condition_key = SHA256(resolved config에서 child seed 계열 제거)
run_key       = SHA256(resolved config 전체)
```

## 3. 전역 시행 정책

### 3.1 Seed

각 child는 `master_seed in 0..9`를 가진다. 난수 역할은 다음처럼 파생한다.

```text
model_init_seed       = master_seed
batch_order_seed      = master_seed + 10_000
dropout_seed          = master_seed + 20_000
negative_sample_seed  = master_seed + 30_000
synthetic_input_seed  = master_seed + 40_000
```

비교 조건에서 같은 master seed는 같은 역할별 난수열을 사용한다. 데이터 split seed는 master seed와 분리한다.

### 3.2 데이터 및 수치형

- dtype은 recipe에 명시한다. 1권 NumPy 재현 recipe는 float64, 2권의 embedding/RNN/Seq2seq recipe는 float32를 사용한다. e09는 atomic override를 따른다.
- dataset source file, 전처리 설정, subset index와 split index의 SHA-256을 기록한다.
- MNIST subset은 shuffle 전 공식 학습 배열의 앞 1,000개 또는 300개를 사용한다.
- Seq2seq split seed는 `1984`로 고정한다.
- PTB vocabulary는 train split에서 생성한 공식 vocabulary로 고정한다.

### 3.3 Paired execution

- 동일 seed의 비교 조건은 동일한 초기 표준정규 weight 원본을 가능한 범위에서 공유한다.
- 모델 구조가 같으면 minibatch index sequence도 공유한다.
- 구조가 다른 경우에도 dataset order seed는 동일하게 유지한다.
- 한 조건만 실패하더라도 대응 seed의 다른 조건을 삭제하거나 재추첨하지 않는다.

### 3.4 평가 및 체크포인트

- training metric과 evaluation metric의 step 기준을 분리한다.
- final test는 지정된 evaluation mode에서 전체 test set을 사용한다.
- `LM-BETTER`는 best validation PPL checkpoint를 저장하고 해당 checkpoint로 test를 1회 평가한다.
- `CNN-DEEP`은 모든 child의 final checkpoint를 저장해 e09가 참조하도록 한다.
- 기타 run은 resolved config, metrics CSV, 요약 JSON, 환경 정보와 최종 또는 best checkpoint를 저장한다.

### 3.5 실패 및 재시도

- NaN/Inf loss, OOM, uncaught exception, metric schema 누락을 실패로 정의한다.
- 동일 run key의 `FINISHED` run이 있으면 재사용한다.
- `FAILED/KILLED` run은 새 attempt를 만들되 `retry_of` 태그를 기록한다.
- 실행기가 자동으로 hyperparameter를 변경해 복구해서는 안 된다.

## 4. 데이터셋 Registry

| Dataset ID | Source | 규모 | 전처리·표현 | 고정 정책 |
| --- | --- | --- | --- | --- |
| DS-MNIST-FLAT | MNIST official train/test | train 60,000; test 10,000 | float input normalized to `[0,1]`, flatten `(784,)`, integer labels | official split fixed; log source checksum |
| DS-MNIST-IMG | MNIST official train/test | train 60,000; test 10,000 | float input normalized to `[0,1]`, shape `(1,28,28)`, integer labels | official split fixed; log source checksum |
| DS-MNIST-1000 | MNIST official train 앞 1,000 | train 1,000; official test retained | same as DS-MNIST-FLAT | subset indices `0..999` fixed |
| DS-MNIST-300 | MNIST official train 앞 300 | train 300; official test retained | same as DS-MNIST-FLAT | subset indices `0..299` fixed |
| DS-SYNTH-ACT | 합성 표준정규 입력 | 1,000×100 | `N(0,1)`, float64 | child input seed 사용; 조건 간 같은 seed의 입력 공유 |
| DS-PTB-W2V | Penn Treebank train | 공식 train corpus; vocab from train | lower-level loader의 token IDs, window 5 contexts | 파일 SHA-256과 vocab hash 기록 |
| DS-PTB-LM | Penn Treebank train/valid/test | 공식 세 split; vocab 10,000 | next-token `(x_t, x_{t+1})` | split·vocab 고정; state reset 시점 명시 |
| DS-SEQ-ADD | `addition.txt` | 전체 shuffle 후 90/10 | character IDs; source/target fixed length | loader split seed `1984` 고정 |
| DS-SEQ-DATE | `date.txt` | 전체 shuffle 후 90/10 | character IDs; source/target fixed length | loader split seed `1984` 고정 |

## 5. Recipe Registry

| Recipe ID | Dataset | 모델 | 완전한 모델 설정 | 완전한 훈련·평가 설정 | Atomic override |
| --- | --- | --- | --- | --- | --- |
| RC-TOY | DS 없음 | analytic objective | `f=x²/20+y²`; grad=`(x/10,2y)`; init=`(-7,2)`; dtype=float64 | 30 updates; float64 | optimizer override |
| RC-MLP | DS-MNIST-FLAT | MLP | `784-[100,100,100,100]-10`; ReLU; SoftmaxCE; bias 0 | batch128; 2,000 updates; sampling with replacement; no BN/dropout/WD; dtype=float64 | optimizer, initializer |
| RC-ACT | DS-SYNTH-ACT | forward probe MLP | 5 hidden layers; width100; bias0; no output loss | no training; save each layer activation; dtype=float64 | activation, initializer |
| RC-BN | DS-MNIST-1000 | MLP+optional BN | `784-[100×5]-10`; Affine→BN(optional)→ReLU; BN γ=1 β=0 momentum=.9 eps=1e-6 | SGD lr=.01; batch100; 20 epochs; CE; dtype=float64 | BN flag, fixed init scale |
| RC-REG | DS-MNIST-300 | MLP regularization | `784-[100×6]-10`; ReLU; He; dropout after each hidden ReLU | SGD lr=.01; batch100; 301 epochs; CE; dtype=float64 | weight_decay_lambda, dropout_ratio |
| RC-CNN-S | DS-MNIST-IMG | SimpleConvNet | Conv30 5×5 s1 p0→ReLU→Pool2→FC100→ReLU→FC10; init std=.01 | Adam lr=.001 β1=.9 β2=.999; batch100; 20 epochs; CE; dtype=float64 | none |
| RC-CNN-D | DS-MNIST-IMG | DeepConvNet | Conv channels 16,16,32,32,64,64; 3×3; pool after conv2/4/6; FC50; dropout .5 after FC and output affine; He | Adam lr=.001 β1=.9 β2=.999; batch100; 20 epochs; CE; dtype=float64 | none |
| RC-DTYPE | DS-MNIST-IMG test | derived inference | source=`CNN-DEEP-sXX`; cast input+params to dtype; eval mode | batch100; accuracy full test; latency warmup20+measure100; dtype=atomic override | dtype |
| RC-W2V | DS-PTB-W2V | CBOW/SkipGram | embedding100; window5; W_in/W_out normal std=.01 | Adam lr=.001; batch100; 10 epochs; dtype=float32 | architecture, objective |
| RC-LM | DS-PTB-LM | 1-layer RNNLM | embedding100; hidden100; TimeEmbedding→RNN/LSTM→TimeAffine→Softmax | SGD lr=20; batch20; BPTT35; 4 epochs; dtype=float32 | cell, max_grad |
| RC-BETTER-LM | DS-PTB-LM | BetterRnnlm | embedding650; 2×LSTM650; dropout .5 at embedding/inter-layer/output; weight tying | SGD lr=20; batch20; BPTT35; clip.25; max40 epochs; valid non-improve→lr/4; best checkpoint; dtype=float32 | none |
| RC-SEQ-ADD | DS-SEQ-ADD | Seq2seq family | char embedding16; hidden128; encoder/decoder LSTM; teacher forcing train; greedy eval | Adam lr=.001; batch128; 25 epochs; clip5; dtype=float32 | architecture, reverse |
| RC-SEQ-DATE | DS-SEQ-DATE | Seq2seq family | char embedding16; hidden256; encoder/decoder LSTM; teacher forcing train; greedy eval | Adam lr=.001; batch128; 10 epochs; clip5; reverse=true; dtype=float32 | architecture |

## 5.1 Toy optimizer

| Atomic run ID | Recipe | Resolved override |
| --- | --- | --- |
| TOY-SGD | RC-TOY | optimizer=SGD; lr=.95 |
| TOY-MOM | RC-TOY | optimizer=Momentum; lr=.1; momentum=.9 |
| TOY-ADAGRAD | RC-TOY | optimizer=AdaGrad; lr=1.5; eps=1e-7 |
| TOY-ADAM | RC-TOY | optimizer=Adam; lr=.3; beta1=.9; beta2=.999; eps=1e-7 |

## 5.2 MNIST MLP optimizer·initializer

| Atomic run ID | Recipe | Optimizer | Initializer |
| --- | --- | --- | --- |
| MLP-SGD-HE | RC-MLP | SGD lr=.01 | He=sqrt(2/fan_in) |
| MLP-MOM-HE | RC-MLP | Momentum lr=.01 momentum=.9 | He=sqrt(2/fan_in) |
| MLP-ADAGRAD-HE | RC-MLP | AdaGrad lr=.01 eps=1e-7 | He=sqrt(2/fan_in) |
| MLP-ADAM-HE | RC-MLP | Adam lr=.001 beta1=.9 beta2=.999 eps=1e-7 | He=sqrt(2/fan_in) |
| MLP-SGD-XAVIER | RC-MLP | SGD lr=.01 | Xavier=sqrt(1/fan_in) |
| MLP-SGD-STD001 | RC-MLP | SGD lr=.01 | Normal std=.01 |

## 5.3 Activation probe

| Atomic run ID | Recipe | Activation | Initializer | Scale |
| --- | --- | --- | --- | --- |
| ACT-SIG-STD1 | RC-ACT | sigmoid | Normal std=1 | 1.0 |
| ACT-SIG-STD001 | RC-ACT | sigmoid | Normal std=.01 | .01 |
| ACT-SIG-XAVIER | RC-ACT | sigmoid | Xavier | sqrt(1/fan_in) |
| ACT-SIG-HE | RC-ACT | sigmoid | He | sqrt(2/fan_in) |
| ACT-TANH-STD1 | RC-ACT | tanh | Normal std=1 | 1.0 |
| ACT-TANH-STD001 | RC-ACT | tanh | Normal std=.01 | .01 |
| ACT-TANH-XAVIER | RC-ACT | tanh | Xavier | sqrt(1/fan_in) |
| ACT-TANH-HE | RC-ACT | tanh | He | sqrt(2/fan_in) |
| ACT-RELU-STD1 | RC-ACT | relu | Normal std=1 | 1.0 |
| ACT-RELU-STD001 | RC-ACT | relu | Normal std=.01 | .01 |
| ACT-RELU-XAVIER | RC-ACT | relu | Xavier | sqrt(1/fan_in) |
| ACT-RELU-HE | RC-ACT | relu | He | sqrt(2/fan_in) |

## 5.4 BatchNorm × initialization scale

| Atomic run ID | Recipe | BatchNorm | weight_init_scale |
| --- | --- | --- | --- |
| BN-OFF-01 | RC-BN | false | 1 |
| BN-ON-01 | RC-BN | true | 1 |
| BN-OFF-02 | RC-BN | false | 0.541169527 |
| BN-ON-02 | RC-BN | true | 0.541169527 |
| BN-OFF-03 | RC-BN | false | 0.292864456 |
| BN-ON-03 | RC-BN | true | 0.292864456 |
| BN-OFF-04 | RC-BN | false | 0.158489319 |
| BN-ON-04 | RC-BN | true | 0.158489319 |
| BN-OFF-05 | RC-BN | false | 0.0857695899 |
| BN-ON-05 | RC-BN | true | 0.0857695899 |
| BN-OFF-06 | RC-BN | false | 0.0464158883 |
| BN-ON-06 | RC-BN | true | 0.0464158883 |
| BN-OFF-07 | RC-BN | false | 0.0251188643 |
| BN-ON-07 | RC-BN | true | 0.0251188643 |
| BN-OFF-08 | RC-BN | false | 0.0135935639 |
| BN-ON-08 | RC-BN | true | 0.0135935639 |
| BN-OFF-09 | RC-BN | false | 0.00735642254 |
| BN-ON-09 | RC-BN | true | 0.00735642254 |
| BN-OFF-10 | RC-BN | false | 0.00398107171 |
| BN-ON-10 | RC-BN | true | 0.00398107171 |
| BN-OFF-11 | RC-BN | false | 0.00215443469 |
| BN-ON-11 | RC-BN | true | 0.00215443469 |
| BN-OFF-12 | RC-BN | false | 0.0011659144 |
| BN-ON-12 | RC-BN | true | 0.0011659144 |
| BN-OFF-13 | RC-BN | false | 0.000630957344 |
| BN-ON-13 | RC-BN | true | 0.000630957344 |
| BN-OFF-14 | RC-BN | false | 0.000341454887 |
| BN-ON-14 | RC-BN | true | 0.000341454887 |
| BN-OFF-15 | RC-BN | false | 0.00018478498 |
| BN-ON-15 | RC-BN | true | 0.00018478498 |
| BN-OFF-16 | RC-BN | false | 0.0001 |
| BN-ON-16 | RC-BN | true | 0.0001 |

## 5.5 Regularization

| Atomic run ID | Recipe | weight_decay_lambda | dropout_ratio | 정책 |
| --- | --- | --- | --- | --- |
| REG-BASE | RC-REG | 0 | 0 | 공통 baseline |
| REG-WD-1E4 | RC-REG | 1e-4 | 0 | L2는 W에만 적용; loss에 .5*λ*sum(W²), grad에 λW |
| REG-WD-1E3 | RC-REG | 1e-3 | 0 | 동일 |
| REG-WD-1E2 | RC-REG | 1e-2 | 0 | 동일 |
| REG-WD-1E1 | RC-REG | 1e-1 | 0 | 동일 |
| REG-DO-01 | RC-REG | 0 | .1 | book-compatible dropout: train mask, eval multiply (1-p) |
| REG-DO-02 | RC-REG | 0 | .2 | 동일 |
| REG-DO-03 | RC-REG | 0 | .3 | 동일 |
| REG-DO-05 | RC-REG | 0 | .5 | 동일 |

## 5.6 CNN training and dtype evaluation

| Atomic run ID | Recipe | Resolved override | Dependency |
| --- | --- | --- | --- |
| CNN-SIMPLE | RC-CNN-S | none | none |
| CNN-DEEP | RC-CNN-D | none | none |
| DTYPE-F64 | RC-DTYPE | dtype=float64 | source child CNN-DEEP with same seed |
| DTYPE-F32 | RC-DTYPE | dtype=float32 | source child CNN-DEEP with same seed |
| DTYPE-F16 | RC-DTYPE | dtype=float16 | source child CNN-DEEP with same seed |

## 5.7 Word embedding

| Atomic run ID | Recipe | Architecture | Objective | Objective details |
| --- | --- | --- | --- | --- |
| W2V-CBOW-NS | RC-W2V | CBOW | negative sampling | sample_size=5; power=.75; shared W_out |
| W2V-SG-NS | RC-W2V | SkipGram | negative sampling | 2*window=10 loss terms; each sample_size=5; power=.75 |
| W2V-CBOW-FULL | RC-W2V | CBOW | full softmax | Time/Batch MatMul to vocab logits; SoftmaxCE |

## 5.8 PTB language model

| Atomic run ID | Recipe | Cell/model | max_grad |
| --- | --- | --- | --- |
| LM-RNN-C025 | RC-LM | vanilla RNN | .25 |
| LM-LSTM-NONE | RC-LM | LSTM | None |
| LM-LSTM-C010 | RC-LM | LSTM | .10 |
| LM-LSTM-C025 | RC-LM | LSTM | .25 |
| LM-LSTM-C050 | RC-LM | LSTM | .50 |
| LM-LSTM-C100 | RC-LM | LSTM | 1.0 |
| LM-LSTM-C500 | RC-LM | LSTM | 5.0 |
| LM-BETTER | RC-BETTER-LM | BetterRnnlm | .25 |

## 5.9 Addition Seq2seq

| Atomic run ID | Recipe | Architecture | reverse |
| --- | --- | --- | --- |
| SEQA-VAN-FWD | RC-SEQ-ADD | Vanilla Seq2seq | false |
| SEQA-VAN-REV | RC-SEQ-ADD | Vanilla Seq2seq | true |
| SEQA-PEEKY-REV | RC-SEQ-ADD | Peeky Seq2seq | true |

## 5.10 Date Seq2seq

| Atomic run ID | Recipe | Architecture | reverse |
| --- | --- | --- | --- |
| SEQD-VAN-REV | RC-SEQ-DATE | Vanilla Seq2seq | true |
| SEQD-PEEKY-REV | RC-SEQ-DATE | Peeky Seq2seq | true |
| SEQD-ATTN-REV | RC-SEQ-DATE | Attention Seq2seq | true |

## 6. 모델별 세부 구현 계약

### 6.1 MLP 공통

- 각 hidden layer: `Affine -> optional BatchNorm -> ReLU -> optional Dropout`.
- output: `Affine -> SoftmaxWithLoss`.
- He=`sqrt(2/fan_in)`, Xavier=`sqrt(1/fan_in)`.
- L2 penalty는 bias·BatchNorm parameter에 적용하지 않고 W에만 적용한다.
- L2 loss는 `.5 * lambda * sum(W**2)`, gradient 추가항은 `lambda * W`다.
- Dropout은 책 호환 방식으로 training에서 binary mask를 곱하고 inference에서 `(1-p)`를 곱한다.
- BatchNorm은 hidden affine 출력에 적용한 후 ReLU를 통과시킨다. `gamma=1`, `beta=0`, `momentum=.9`, `eps=1e-6`.

### 6.2 CNN

- SimpleConvNet: `Conv(30,5x5,s1,p0)-ReLU-Pool(2,s2)-Affine(100)-ReLU-Affine(10)`.
- DeepConvNet: `Conv16-Conv16-Pool-Conv32-Conv32-Pool-Conv64-Conv64-Pool-Affine50-ReLU-Dropout(.5)-Affine10-Dropout(.5)`.
- DeepConvNet convolution은 3×3을 사용하며 공식 pad 구성은 `[1,1,1,2,1,1]`, stride는 모두 1이다.

### 6.3 Word2Vec

- CBOW hidden은 `2*window` context embedding의 평균이다.
- Skip-gram은 target embedding으로 각 context position을 예측하고 10개 loss를 합산한다.
- e10 비교용 loss는 raw 합이 아니라 예측 context term당 loss로 정규화한다.
- full softmax 조건도 one-hot toy 구현이 아니라 동일 PTB integer-index embedding 모델에서 output objective만 교체한다.

### 6.4 Language model

- stateful recurrent layer를 사용하며 epoch/validation/test 경계에서 state를 reset한다.
- perplexity는 token-level mean cross entropy의 exponential로 계산한다.
- gradient clipping은 모든 parameter gradient의 global L2 norm에 적용한다.
- clipping 전 norm, scale factor와 clipping 여부를 매 update 기록하거나 집계 가능하게 보존한다.

### 6.5 Seq2seq

- train은 teacher forcing, evaluation은 greedy autoregressive decode를 사용한다.
- sequence exact-match는 시작 문자를 제외한 생성 문자열 전체가 정답과 일치할 때 1이다.
- reverse는 source sequence만 뒤집으며 target은 변경하지 않는다.
- Attention run은 test 대표 샘플의 attention matrix를 artifact로 저장한다.

## 7. MLflow 기록 계약

### 7.1 Parent tags

```text
run_type=condition_parent
atomic_run_id=<registry id>
recipe_id=<recipe id>
condition_key=<sha256>
expected_children=10
runner_schema_version=1
```

### 7.2 Child tags/params

```text
run_type=seed_trial | derived_evaluation
atomic_run_id
condition_key
run_key
master_seed
source_run_id         # derived evaluation only
code.git_commit
code.dirty
dataset.digest
```

모델·optimizer·훈련·데이터 설정은 child에도 flatten된 params로 모두 기록한다. parent만 읽어야 설정을 알 수 있는 구조를 금지한다.

### 7.3 최소 metric schema

```text
train.loss
train.accuracy                 # applicable task only
valid.loss
valid.accuracy | valid.ppl
test.accuracy | test.ppl
system.epoch_seconds
system.samples_per_second
grad.global_norm_before_clip   # applicable run only
grad.clip_fraction             # applicable run only
```

### 7.4 최소 artifact schema

```text
config/resolved.json
environment/python.txt
environment/packages.txt
environment/git.json
data/dataset_manifest.json
metrics/history.csv
summary/final.json
checkpoints/best_or_final.*
plots/                         # optional per child; required at aggregate
```

## 8. 분석 Registry

| Analysis ID | 조회 atomic run | 사용 구간 |
| --- | --- | --- |
| e01 | TOY-SGD, TOY-MOM, TOY-ADAGRAD, TOY-ADAM | all steps |
| e02 | MLP-SGD-HE, MLP-MOM-HE, MLP-ADAGRAD-HE, MLP-ADAM-HE | updates 0..1999 |
| e03 | ACT-* | layers 1..5 |
| e04 | MLP-SGD-STD001, MLP-SGD-XAVIER, MLP-SGD-HE | updates 0..1999 |
| e05 | BN-OFF-01..16, BN-ON-01..16 | epochs 0..19 |
| e06 | REG-BASE, REG-WD-1E4, REG-WD-1E3, REG-WD-1E2, REG-WD-1E1 | epochs 0..200 |
| e07 | REG-BASE, REG-DO-01, REG-DO-02, REG-DO-03, REG-DO-05 | epochs 0..300 |
| e08 | CNN-SIMPLE, CNN-DEEP | epochs 0..19 + full test |
| e09 | DTYPE-F64, DTYPE-F32, DTYPE-F16 | full test + benchmark samples |
| e10 | W2V-CBOW-NS, W2V-SG-NS | epochs 0..9 |
| e11 | W2V-CBOW-FULL, W2V-CBOW-NS | epochs 0..9 |
| e12 | LM-RNN-C025, LM-LSTM-C025 | epochs 0..3 + test PPL |
| e13 | LM-LSTM-NONE, LM-LSTM-C010, LM-LSTM-C025, LM-LSTM-C050, LM-LSTM-C100, LM-LSTM-C500 | epochs 0..3 |
| e14 | LM-LSTM-C025, LM-BETTER | best-valid checkpoint + test PPL |
| e15 | SEQA-VAN-FWD, SEQA-VAN-REV | epochs 0..24 |
| e16 | SEQA-VAN-REV, SEQA-PEEKY-REV | epochs 0..24 |
| e17 | SEQD-VAN-REV, SEQD-PEEKY-REV, SEQD-ATTN-REV | epochs 0..9 |

## 9. 실행 전 Validation Checklist

- atomic run ID가 유일한가.
- recipe와 override를 합친 후 미정 필드가 없는가.
- dataset digest와 split digest가 생성되었는가.
- parent condition key가 기존 FINISHED parent와 충돌하지 않는가.
- child 0..9 중 기존 FINISHED run을 제외한 missing seed만 plan에 남았는가.
- 비교 조건의 seed 정책과 batch-order 정책이 동일한가.
- derived run의 source checkpoint가 존재하고 dtype cast가 원본을 변형하지 않는가.
- metric name과 step semantics가 recipe schema와 일치하는가.

## 10. 공식 코드 기준 파일

- 1권: `common/multi_layer_net.py`, `common/multi_layer_net_extend.py`, `common/layers.py`, `common/optimizer.py`
- 1권 실험: `ch06/*`, `ch07/train_convnet.py`, `ch08/train_deepnet.py`, `ch08/half_float_network.py`
- 2권: `common/trainer.py`, `common/optimizer.py`, `common/time_layers.py`, `common/util.py`
- 2권 실험: `ch04/train.py`, `ch05/train.py`, `ch06/train_rnnlm.py`, `ch06/train_better_rnnlm.py`, `ch07/train_seq2seq.py`, `ch08/train.py`
