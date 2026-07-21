# deepscratch2 원본 대비 실험·분석 코드 차이

검증일: 2026-07-21

## 결론

`deepscratch2`는 원본 `01_deep-learning-from-base/deep-learning-from-scratch-2`의
알고리즘과 주요 하이퍼파라미터를 상당 부분 옮긴 **확장 재현 실험군**이다. 그러나 원본
스크립트와 동일한 실행·그래프 재현물은 아니다. 따라서 원본 그림 또는 수치를 그대로
재현했다고 해석해서는 안 된다.

- 원본 ch01--ch02에는 현재 대응하는 `deepscratch2` 실험이 없다.
- e01--e05은 원본 ch03--ch08의 주제를 포괄하지만, 10개 seed 반복, GPU/CuPy,
  MLflow, 추가 비교 조건과 집계 그래프를 사용한다.
- 모델 계열, PTB/문자 데이터셋, 핵심 크기와 optimizer는 대부분 원본과 맞는다.
- e01은 원본의 합산 loss와 기록 주기를 복원했지만, 고성능 alias sampler와 seed 집계를
  유지한다. 따라서 원본보다 의도에 가깝게 실행되지만 완전한 코드 재현은 아니다.

판정 기준은 다음과 같다.

| 판정        | 의미                                                                                               |
| ----------- | -------------------------------------------------------------------------------------------------- |
| 일치        | 원본의 해당 기본 조건과 데이터·구조·주요 학습 설정이 동일하다.                                     |
| 부분 일치   | 기본 조건은 유지하지만, 관측/실행 방식 또는 일부 조건이 달라 원본 수치·곡선을 직접 비교할 수 없다. |
| 불일치/확장 | 원본에 없는 비교이거나, 데이터·목적함수·핵심 학습 설정이 달라 원본 재현으로 볼 수 없다.            |

## 비교 범위와 방법

비교 대상은 현재의 `experiments/deepscratch2/config/`,
`experiments/deepscratch2/analysis/`, 실행기
`src/mlprosection/experiment/executors/sequence.py`, 그리고 원본 ch03--ch08의
`train*.py`, 모델 및 분석 스크립트다. 원본 ch01--ch02는 e01--e05 카탈로그에 대응
항목이 없으므로 범위 밖으로 기록했다.

`deepscratch1-original-setting-differences.md`의 방식을 따라, 모델/학습 조건과
그래프·관측 조건을 분리해 평가했다. 단위 테스트는 실행 경로가 실제 원본과 같은 결과를
낸다는 증명은 아니며, 구성과 최소 모델 동작의 회귀 검사다.

## 실험별 판정

| 현재 실험                             | 원본 대응                                                       | 판정        | 근거                                                                                                                                                                                                                                                                                                       |
| ------------------------------------- | --------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| e01 CBOW/Skip-gram negative sampling  | ch04 `train.py`, `cbow.py`, `skip_gram.py`                      | 부분 일치   | PTB train, window=5, embedding=100, batch=100, Adam, 10 epoch, negative 5개, power=0.75은 일치한다. 전용 `BookWord2VecTrainer`가 원본의 합산 loss, partial-batch discard, epoch shuffle 및 첫 update부터 20 iteration 기록을 따른다. 원본은 CBOW 한 조건만 기본 실행하고 sampler·seed 집계도 다르다. |
| e02 full softmax vs negative sampling | ch03 `train.py` 및 ch04 `train.py`                              | 불일치/확장 | 원본 full softmax는 `You say goodbye ...` toy corpus, window=1, hidden=5, batch=3, 1,000 epoch의 `SimpleCBOW`다. 현재는 PTB/window=5/embedding=100/10 epoch CBOW에 full softmax를 적용하고 e01의 NS와 비교한다. 이 비교 자체는 원본에 없다.                                                                |
| e03 RNNLM 비교                        | ch05 `train.py`, ch06 `train_rnnlm.py`, `train_better_rnnlm.py` | 부분 일치   | `LM-LSTM-C025`와 `LM-BETTER`는 각각 ch06의 LSTM/BetterRnnlm 크기, batch=20, time=35, SGD lr=20, max_grad=.25, 4/40 epoch 및 Better의 0.25 lr decay를 따른다. 반면 `LM-RNN-C025`는 ch05 SimpleRnnlm 구조이지만, 원본 ch05의 첫 1,000 token, batch=10, time=5, lr=.1, 100 epoch 대신 ch06 조건으로 학습한다. |
| e04 addition seq2seq                  | ch07 `train_seq2seq.py`                                         | 부분 일치   | 기본 vanilla forward 조건은 addition, embedding=16, hidden=128, batch=128, Adam, 25 epoch, max_grad=5로 일치한다. 원본은 vanilla forward 하나를 기본 실행하며 reverse/Peeky는 수동 선택지다. 현재는 forward/reverse, Peeky, 그리고 원본 ch07에 없는 Attention 조건까지 5개를 함께 비교한다.                |
| e05 date attention seq2seq            | ch08 `train.py`, `visualize_attention.py`                       | 부분 일치   | Attention 기본 조건은 date, 입력 반전, embedding=16, hidden=256, batch=128, Adam, 10 epoch, max_grad=5로 일치한다. 현재는 vanilla/Peeky 조건을 추가 비교하고, seed별 split·집계와 attention 시각화 방식이 원본과 다르다.                                                                                   |

## 파라미터 불일치

이 표는 데이터, 모델 크기, optimizer, 학습 길이처럼 실행 결과를 직접 바꾸는 설정만
다룬다. 그래프를 어떻게 그리고 어떤 조건을 한 그림에 넣는지는 다음 절에서 분리한다.

| 실험/조건 | 원본 대비 파라미터 상태 | 조치 |
| --- | --- | --- |
| e01 CBOW, Skip-gram | PTB train, window=5, embedding=100, batch=100, Adam `.001`, 10 epoch, negative=5, power=.75가 맞는다. e01은 이제 합산 loss도 맞춘다. 단, alias 복원 sampler는 원본 CPU의 비복원 sampler와 다르다. | alias sampler는 원본의 성능 결함을 피하는 프레임워크 구현으로 유지하고, 결과에 명시한다. |
| e02 CBOW full softmax | 대응 원본 ch03은 toy corpus/window=1/hidden=5/batch=3/1,000 epoch이고, 현재는 PTB/window=5/embedding=100/batch=100/10 epoch다. | 책 재현용이라면 ch03 toy 조건을 별도 추가한다. 현재 조건은 PTB 확장 실험으로 유지한다. |
| e03 LM-RNN-C025 | ch05와 달리 전체 PTB, batch=20, time=35, lr=20, max_grad=.25, 4 epoch를 쓴다. 원본은 첫 1,000 token, batch=10, time=5, lr=.1, 100 epoch다. | ch05 baseline을 별도 조건으로 추가한다. |
| e03 LM-LSTM-C025 | ch06 `train_rnnlm.py`의 batch=20, time=35, size=100, SGD lr=20, max_grad=.25, 4 epoch와 맞는다. | 유지. |
| e03 LM-BETTER | ch06 `train_better_rnnlm.py`의 size=650, dropout=.5, batch=20, time=35, lr=20, max_grad=.25, 40 epoch와 맞는다. | 유지. |
| e04 vanilla forward | addition, embedding=16, hidden=128, batch=128, Adam, 25 epoch, max_grad=5가 맞는다. split은 legacy NumPy 1984 permutation으로 고정했다. | 유지. 모델 초기화·batch 순서만 trial seed별로 달라진다. |
| e05 attention | date, reverse, embedding=16, hidden=256, batch=128, Adam, 10 epoch, max_grad=5가 맞는다. split은 legacy NumPy 1984 permutation으로 고정했다. | 유지. 모델 초기화·batch 순서만 trial seed별로 달라진다. |

## 실험 설계 불일치와 병합 기준

여기서 병합은 **같은 데이터셋에서 하나의 비교축만 의도적으로 바꾼 조건**을 하나의
실험군/분석 그림으로 관리한다는 뜻이다. 이 기준을 만족하지 않으면, 함께 보여 줄 수는
있어도 통제된 단일 비교 실험으로 이름 붙이지 않는다.

| 실험군 | 데이터셋·비교축 | 병합 판단 | 그래프 규칙 |
| --- | --- | --- | --- |
| e01 CBOW/Skip-gram | PTB train; architecture `{CBOW, Skip-gram}` | 병합 가능. 원본 코드에서 Skip-gram은 주석 처리된 선택지다. | 각 조건의 책형 raw-loss curve는 분리하고, 둘을 비교하는 확장 그래프는 normalized loss를 쓴다. |
| e02 full softmax/NS | PTB train; output objective `{full softmax, NS}` | 병합 가능하지만 책 밖의 확장 실험이다. | 두 조건은 같은 logging/profiling 설정을 써야 하며, loss 절대값이 아니라 속도·메모리·별도 품질 지표를 비교한다. |
| e03 RNN/LSTM/BetterRnnlm | 모두 PTB이지만 모델 크기·학습 길이·scheduler가 함께 다르다. | 단일 통제 실험으로는 병합 불가. | ch05 RNN, ch06 LSTM, ch06 BetterRnnlm을 각각 책형 baseline으로 두고, 세 모델 overview는 보조 그림으로만 둔다. |
| e04 Seq2seq family | addition; architecture와 input reverse의 2축 | 병합 가능. 원본 vanilla와 주석 처리된 reverse/Peeky, 다음 장 Attention을 명시적 factorial 확장으로 관리한다. | 원본 vanilla-forward 단일 curve를 기준 그림으로 두고, 2축 비교는 별도 확장 그림으로 둔다. |
| e05 Seq2seq family | date; architecture `{vanilla, Peeky, Attention}` | 병합 가능. 원본 Attention과 주석 처리된 vanilla/Peeky가 같은 데이터·입력 반전 조건을 쓴다. | Attention 단일 curve/5개 labelled map을 기준 그림으로 두고, 세 모델 비교는 확장 그림으로 둔다. |

## 결과에 영향을 주는 핵심 차이

### e01/e02 word2vec 목적함수와 sampler

원본 ch04 `NegativeSamplingLoss`는 positive 1개와 negative 5개의 sigmoid loss를
**합산**한다. e01은 `loss.reduction: sum`과 `BookWord2VecTrainer`를 사용하여 이
gradient scale을 복원한다. Skip-gram도 context 10개의 loss를 합산하므로 원본과 같은
목적함수 단위를 쓴다. 호환성을 위해 `train/normalized_loss`도 함께 저장하지만, e01
원본형 그래프는 `train/book_loss`를 사용한다.

e02의 full-softmax 조건은 현재의 평균 cross-entropy를 유지한다. 따라서 e02에서
full-softmax와 negative-sampling의 **loss 절대값**은 여전히 같은 척도가 아니며,
그 비교는 속도·메모리·학습 경향 중심으로 해석해야 한다.

원본 CPU sampler는 positive target을 제외하고 배치의 negative 5개를 **비복원**으로
추출한다. 현재 `UnigramSampler`는 alias-table 기반 **복원** 추출 후 target만 고정
round rejection으로 제외한다. negative끼리 중복될 수 있으며 난수열도 다르다.

원본 CPU 구현에는 성능상 중대한 문제가 있다. `get_negative_sample()`이 batch의 각
target마다 vocabulary 전체 확률벡터를 `copy()`하고, positive 확률을 0으로 만든 뒤 다시
정규화하여 `np.random.choice()`를 호출한다. 즉 sample 5개를 얻는 과정이 사실상
`O(batch_size * vocab_size)`이며, 이 비용이 full softmax의 고도로 최적화된 dense matrix
multiplication보다 커질 수 있다. 이것이 실제 실행에서 negative sampling이 더 빠르다는
기대와 반대로 더 느려질 수 있는 원인이다. 현재 e01의 alias table은 corpus 분포를 한 번만
구축하고 device에서 draw하므로 이 병목을 피하는 의도된 변경이다.

중요하게도 책 원본의 GPU 분기는 속도를 위해 `replace=True`로 바로 추출하며 positive
target을 제외하지 않는다. 따라서 같은 단어가 positive와 negative label 양쪽에 들어가
상반된 gradient를 받는 결함이 있다. e01의 현재 sampler는 target rejection을 수행하므로
이 원본 GPU 분기의 결함은 피한다. 따라서 e01은 원본 **CPU** 동작과는 비복원 여부가,
원본 **GPU** 동작과는 positive 제외 여부가 다르며, 이 차이는 현 구현 쪽의 의도된 수정이다.

또한 현재 sampler의 고정 횟수 rejection은 일반적으로는 결함이다. 다섯 번의 draw 뒤에도
target이 남으면 `(target + 1) % vocab_size`를 강제로 넣으므로, 이는 target을 제외한
unigram 분포에서 뽑은 값이 아니다. target 확률을 `p`라 하면 이 fallback 확률은 `p^5`이고
그 확률 질량이 다음 token id로 쏠린다. e01 PTB의 power=.75 분포에서는 최대 `p`가 약
0.01769이어서 이 fallback의 상한은 약 `1.73e-9`로 실질 영향은 매우 작다. 하지만 작은
vocabulary 또는 한 token이 지배적인 corpus에서는 학습 표본을 크게 왜곡할 수 있다.

### 데이터 분할·난수·수치 환경

원본 sequence loader는 기본 seed 1984의 legacy NumPy `shuffle`로 한 번 90/10 split을
만든다. e04/e05은 `split_seed: 1984`와 `legacy_numpy_randomstate` 옵션으로 동일한
permutation을 사용한다. 모델 초기화와 batch 순서는 trial seed별로 달라진다. 원본은 기본
CPU NumPy(또는 사용자가 주석 해제한 GPU)와 비고정 모델 난수를 사용하지만, 현재는 float32 CuPy
`cuda:0`, 분리 seed stream, deterministic 요청을 기본으로 한다.

이 차이는 재실행 가능성에는 이롭지만 원본의 한 번 실행과 bit-for-bit 또는 동일 곡선
재현을 불가능하게 한다.

### 추가 조건과 평가

현재 e03은 매 epoch validation/test perplexity를 계산하고, e04/e05은 매 epoch 전체
test exact-match와 token accuracy를 기록한다. 원본 e03의 일반 LSTM은 마지막 test
perplexity만 출력하고, 원본 seq2seq는 exact match만 기록한다. BetterRnnlm의 validation
기반 lr 감소는 원본과 같은 규칙이지만 현재는 다른 e03 조건에도 validation/test를
추가 측정한다.

## 분석 코드 판정

원본은 단일 실행의 `trainer.plot()` 또는 `plt.plot()`을 사용한다. 반면
`experiments/deepscratch2/analysis/common.py`와 `render.py`는 각 atomic run의 최신
완료 seed trial을 모아 mean/min/max error band와 CSV 요약을 만든다. 이 차이는 의도된
확장이지만 원본 그래프와 동일하지 않다.

| 분석              | 원본과의 차이                                                                                                                                                                                      | 영향                                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| e01               | 원본 ch04 trainer는 합산 loss를 첫 update와 이후 20 iteration 간격으로 기록한다. 현재 전용 trainer와 `e01_cbow_skipgram.py`는 같은 기록 규칙의 단일 CBOW seed curve를 그린다.                     | 원본형 기본 그래프다. CBOW--Skip-gram seed 집계 비교는 별도 `e01_cbow_skipgram_comparison.py` 확장 그림으로 분리했다. |
| e02               | 원본에는 PTB full-softmax 대 negative-sampling 공통 그래프가 없다. 현재는 두 목적함수의 normalized loss를 seed별 집계한다.                                                                              | 목적함수 loss의 절대값 비교가 아닌 확장 분석이다.                                                        |
| e03               | 원본은 각 스크립트의 단일 train perplexity curve(일반 LSTM은 ylim 0--500)를 그린다. 현재는 세 모델의 mean/min/max train perplexity를 공통 축에 표시한다.                                           | 비교용 그래프이며 책 그림 재현이 아니다.                                                                 |
| e04/e05           | 원본은 한 조건의 epoch exact-match curve를 그린다. 현재는 여러 모델·입력 방향의 seed 집계 curve를 그린다.                                                                                          | baseline trend 비교에는 쓸 수 있지만 원본 그림과 겹쳐 해석하면 안 된다.                                  |
| e05 attention map | 원본은 seed 1984로 뽑은 test 예 5개를 입력 순서로 되돌리고 문자 tick label을 붙여 표시한다. 현재는 최신 완료 trial 중 첫 run의 첫 test 예 한 개를 표시하며, 입력 순서 복원·문자 tick label이 없다. | alignment의 의미를 읽기 위한 원본 시각화와 동일하지 않다. 또한 대표 run/예가 명시적으로 고정되지 않는다. |

## 유지해도 되는 해석과 후속 조치

현재 결과는 다음 질문에는 적합하다: 같은 현재 실행 환경에서 CBOW/Skip-gram,
full-softmax/negative-sampling, RNN/LSTM/BetterRnnlm, vanilla/Peeky/attention의 상대적
학습 경향과 seed 간 변동을 비교하는 것.

원본 재현을 목표로 한다면 다음을 별도 호환 모드로 구현해야 한다.

1. 필요하면 e01 원본형 그래프를 단일 seed로도 렌더링하고, ch03 toy full-softmax 조건을 e02과 분리한다.
2. e03에 ch05의 1,000-token SimpleRnnlm 조건을 별도 run으로 둔다.
3. e04/e05은 원본 seed 1984 split으로 고정했다. 다음으로 원본 기본 조건과 확장 비교 그림을 분리한다.
4. e05 원본형 분석은 5개 역순/labelled attention map을 사용한다.

## 검증 기록

다음 정적 검증을 통과했다.

```text
uv run pytest tests/trainer/test_book_word2vec_trainer.py \
  tests/experiment/test_deepscratch2_catalog.py \
  tests/nn/test_unigram_sampler.py -q
# 6 passed

uv run python -m experiments.deepscratch2.run_all --dry-run
# e01--e05, 140 planned runs 확인

git diff --check
# 통과
```

이 검증은 카탈로그의 140 seed trial 선언과 word2vec/sequence 모델의 기본 동작을
확인한다. 원본과의 수치 동등성 시험은 위 차이들 때문에 실행하지 않았고, 그 동등성은
현재 설계의 보장 대상도 아니다.
