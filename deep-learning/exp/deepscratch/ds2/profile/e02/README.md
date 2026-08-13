# DS2 e02 Word2Vec profiling

이 디렉터리는 다음 열네 조건을 같은 PTB workload로 비교하고 update 측정값에서
epoch 및 전체 10-epoch 실행 시간을 추정한다.

- 교재 원본 및 adaptation: CBOW/SkipGram 각각 NS, embedding FS, one-hot FS
- `mlprosection` 구현: CBOW/SkipGram 각각 NS, embedding FS, one-hot FS
- `mlprosection` 전용: fused negative-sampling CBOW/SkipGram

교재 e02에 native full-softmax PTB 모델은 없으므로, original FS는 ch04의 embedding
입력부에 ch03의 `MatMul`과 `SoftmaxWithLoss` 출력부를 붙인 adaptation을 사용한다.
original one-hot FS는 ch03처럼 minibatch 입력과 정답을 one-hot으로 변환하고
`MatMul(W_in)`을 수행한다.
CBOW와 SkipGram은 실행 shape가 크게 다르고, NS와 FS는 CPU/GPU에서 병목 특성이
달라 여덟 구현 조건을 모두 측정한다.

구현체 조건은 현재 `Word2VecExecutor`의 모델/objective 실행 경로를 따른다.
SkipGram의 세 조건은 장치와 objective에 관계없이 center를 유지한 grouped tensor를
사용한다. NS는 grouped candidate score를, 두 FS는 center별 full-vocabulary logits와
grouped target loss를 계산한다. one-hot FS는 입력과 grouped target을 minibatch
단위 dense one-hot tensor로 만든다.
프로파일러는 이 실행 경로를 phase별로 나누어 계측한다.

## 단계별 실행

모든 단계는 `--device cpu`와 `--device cuda:0`에서 같은 방식으로 실행한다.
공식 진입점은 다음 하나다.

```bash
just exp profile ds2 -e 02
```

옵션을 생략하면 CPU와 GPU에서 original adaptation 및 구현의 CBOW/SkipGram
조건을 모두 실행한다. original 조건을 먼저 실행하고 각 구현/모델 그룹은
one-hot FS, embedding FS, NS, 지원되는 경우 fused NS 순서다. 각 장치에서 cold update와
steady update 분포 및 연속 throughput window를 측정하고,
epoch/전체 시간 평균과 반복 표준편차 외삽, 가능한 모든 구성요소 측정을 수행한다. 개별
측정값을 순서대로 출력한 뒤 마지막에 장치×CBOW/SkipGram 비교 표를 출력한다.
표의 원본 forward/backward는 objective를 포함한다는 각주가 붙고, 핵심 실행시간
행에서 가장 빠른 값은 굵게 표시한다.

일부만 측정하려면 옵션을 지정한다.

```bash
just exp profile ds2 -e 02 \
  --device cpu \
  --condition implemented-cbow-ns \
  --mode modules \
  --component model-forward \
  --module-iterations 30
```

Typer list 옵션은 조건이나 장치를 여러 번 지정할 수 있다.

| stage | warmup | 측정 | 반복 | phase 측정 | 용도 |
| --- | ---: | ---: | ---: | ---: | --- |
| `update` | 5 | 10 updates | 3 | 없음 | cold/steady 분리와 빠른 추정 |
| `estimate` | 20 | 50 updates | 5 | 없음 | epoch 및 전체 실행 시간 계획용 기본 추정 |
| `detail` | 50 | 200 updates | 5 | 5 updates | 안정된 throughput과 단계별 병목 확인 |

먼저 한 조건을 CPU에서 빠르게 확인한다.

```bash
just exp profile ds2 -e 02 \
  --device cpu \
  --mode update \
  --condition implemented-cbow-ns \
  --update-warmup 5 \
  --measured-updates 10 \
  --update-repetitions 3
```

그다음 열네 조건의 CPU 계획치를 구한다.

```bash
just exp profile ds2 -e 02 \
  --device cpu \
  --mode update \
  --measured-updates 50 \
  --update-repetitions 5
```

GPU도 device만 바꾸어 동일하게 실행한다.

```bash
just exp profile ds2 -e 02 \
  --device cuda:0 \
  --mode all
```

`--warmup-updates`, `--measured-updates`, `--phase-updates`, `--repetitions`를
주면 stage 기본값을 개별적으로 덮어쓴다. `--condition`은 반복 지정할 수 있다.

JSON의 각 조건에는 다음 값이 기록된다.

- `cold_ms_per_update`: workload 생성 직후 첫 synchronized update
- `steady_event_*_ms_per_update`: warmup 뒤 연속 update에 건 CUDA event 분포.
  update 사이에는 synchronize하지 않고 모든 event를 기록한 뒤 한 번만 동기화한다.
- `mean_ms_per_update`: 별도 반복 window에서 측정한 steady throughput 평균
- `stdev_ms_per_update`: 반복 throughput window의 `ms/update` 표준편차
- `updates_per_epoch`: PTB context 수와 batch/drop-last 규칙으로 계산한 값
- `estimated_first_epoch_seconds`: `cold + steady × (updates_per_epoch - 1)`
- `estimated_seconds_total`: `cold + steady × (전체 updates - 1)`
- `estimated_seconds_per_epoch`: 위 전체 추정값을 epoch 수로 나눈 평균
- `estimated_repeat_stdev_seconds_{per_epoch,total}`: 반복 window 사이의
  steady 속도 표준편차를 update 수에 비례해 외삽한 값
- `phase_ms_per_update`, `phase_share`: `detail` 단계에서만 기록되는 세부 비용

전체 비교 보고서는 열네 조건을 담은 결과를 입력으로 생성한다.

```bash
uv run python -m exp.deepscratch.ds2.profile.e02.analyze \
  --input .cache/experiments/deepscratch/ds2/e02/implemented/profile/update_cuda0.json
```

기본 출력은 git에서 제외되는 `.cache/experiments/deepscratch/ds2/e02/implemented/profile/`에 저장된다.

- `update*.json`: 조건별 synchronized throughput, 추정 시간, phase timing
- `modules*.json`: model/objective/optimizer 구성요소별 timing
- `e02_comparisons.csv`: original→implemented, NS→FS 및 embedding→one-hot 비교
- `e02_analysis.md`: 적용 최적화와 모델별/목적함수/input projection별 해석
- `nsys/cuda_api_summary.csv`: 조건별 CUDA kernel launch API 호출 수

전체 throughput 구간은 CPU에서는 no-op, GPU에서는 CUDA stream 동기화를 시작과
끝에서 수행한다. 따라서 GPU 결과는 host enqueue 시간만이 아니라 완료된 device 작업을
포함한다. phase timing은 각 phase 경계도 동기화하므로 작은 커널에 추가 지연을 준다.
속도 비교에는 전체 throughput을, 병목 위치 확인에는 phase 비중을 사용한다.
구현 조건의 detail phase는 `model_forward`, `objective_forward`,
`objective_backward`, `model_backward`를 각각 기록한다. 교재 원본은 loss가 모델
내부에 포함되어 있어 `forward`와 `backward` 단위로만 기록한다.

`cold_ms_per_update`는 새 workload의 첫 batch 비용이므로 GPU context 자체의 최초
생성 비용까지 보장하는 process-cold 값은 아니다. steady event 분포는 update별
변동을 진단하고, 전체 추정에는 event 계측 오버헤드를 피한 연속 throughput window를
사용한다. epoch/전체 값은 checkpoint, epoch shuffle, MLflow 및 artifact 저장 비용을
포함하지 않는 순수 update-path 외삽값이다.
반복 표준편차 외삽은 각 반복 window에서 관측된 steady 속도 차이가 전체 run 동안
유지된다는 가정에 기반한다. 따라서 update별 독립 잡음의 표준편차나 평균의
신뢰구간이 아니라, 반복 실행 사이의 속도 변동을 나타내는 계획용 수치다. 한 번만
측정한 cold update는 고정값으로 취급하므로 cold 변동성은 포함되지 않는다.

## 구성요소별 프로파일

구현 모델은 실제 e02 batch와 tensor shape를 사용하되 각 구성요소를 독립적으로
측정한다. backward에 필요한 forward/cache 준비는 측정 구간 밖에서 매번 다시
수행한다.

```bash
just exp profile ds2 -e 02 \
  --device cpu \
  --mode modules \
  --condition implemented-cbow-ns \
  --module-warmup 5 \
  --module-iterations 20
```

측정 가능한 구성요소는 `batch_adapter`, `objective_prepare`, `model_forward`,
`objective_forward`, `objective_backward`, `model_backward`, `optimizer`이다.
fused NS는 결합 경계인 `fused_forward_loss`와 `fused_backward`를 대신 기록한다.
Negative Sampling 조건의 `objective_prepare`에는 candidate sampling이 포함된다.
`--component model_forward --component model_backward`처럼 일부만 선택할 수 있다.

각 결과에는 warmup 시간과 measured `mean_ms`, `stdev_ms`, min/max, p50/p95가
기록된다. 더 낮은 수준의 특정 연산은 같은 공통 `BenchmarkRunner`에 callable을
등록하는 방식으로 목록을 확장한다.

## Nsight Systems

CUDA kernel 수, launch gap, GEMM과 scatter 비중은 다음으로 수집한다.

```bash
bash exp/deepscratch/ds2/profile/e02/run_nsys.sh
```

각 report에는 condition throughput과 세부 phase를 나타내는 NVTX range가 들어간다.
짧은 trace가 필요하면 `MEASURED_UPDATES=20 PHASE_UPDATES=2`처럼 조절한다.
특정 조건만 다시 수집하려면 condition 이름을 인자로 전달한다.

```bash
bash exp/deepscratch/ds2/profile/e02/run_nsys.sh implemented-skipgram-fs
```

GUI 없이 요약하려면 예를 들어 다음을 실행한다.

```bash
nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  .cache/experiments/deepscratch/ds2/e02/implemented/profile/nsys/implemented-cbow-ns.nsys-rep
```

## 해석할 때 주의할 점

- NS와 FS loss의 수치 스케일은 서로 다르므로 runtime 비교를 품질 비교로 해석하지 않는다.
- 구현체 trainer는 매 update 후 reporting loss를 다시 forward한다. profiler는 이 비용을
  `post_update_loss`로 별도 기록하면서 전체 throughput에는 포함한다.
- 변경 전 e02 `training_time_s`는 `device_timing: false`인 비동기 host window였다.
  현재 e02 재실행은 `synchronize_train: true`로 전체 학습 경계에서 동기화하며,
  과거 run 분석에는 이 profiler의 synchronized throughput을 사용한다.
- SkipGram-NS와 SkipGram-FS adapter는 모두 100개 center와 context label 10개를
  grouped tensor로 처리한다.
- NS는 각 grouped label의 sampled candidate를 평가하고, FS는 100개 center
  logits에 context label 10개를 묶어 처리한다.
- 결과 공유 시 GPU, CuPy/CUDA 버전, batch size와 update 수를 함께 기록한다.
- CPU와 GPU JSON은 서로 덮어쓰지 않도록 `--output` 이름을 구분한다.

## Vocabulary size sweep

구현체 다섯 조건을 대상으로 vocabulary 크기에 따른 NS/Full Softmax 교차점을
측정하려면 반드시 `--vsweap`을 명시한다.

```bash
just exp profile ds2 -e 02 --vsweap
```

기본값은 GPU에서 `V=10k, 25k, 50k, 100k, 250k, 500k, 1M`, CPU에서
`V=1k, 2k, 5k, 10k, 20k, 50k`를 순회한다. `--vocab-size`를 명시하면 모든 선택 장치에
해당 범위를 적용한다.
embedding 100, batch 100, context width 10, negative 5, conditional-CDF sampler,
dense Adam 및 post-update loss 경로는 현재 구현과 동일하다. synthetic uniform
vocabulary를 사용하며, 기본적으로 20 update를 warmup한 뒤 50 update의 개별 분포와
50-update throughput window 5회를 동기화해 측정한다.

범위를 줄이거나 장치를 바꾸려면 옵션을 반복 지정한다.

```bash
just exp profile ds2 -e 02 --vsweap \
  --device cuda:0 \
  --vocab-size 10000 \
  --vocab-size 50000 \
  --vocab-size 100000
```

결과는 장치별 `results/<device>/vsweap.json`에 저장된다. 같은 디렉터리의
`vsweap.png`는 공통 Matplotlib 테마로 CBOW/SkipGram subplot을 만들고, vocabulary
size를 x축(log scale), update당 시간을 y축(ms)으로 표시하며 95% 신뢰구간 band를
함께 그린다. 표준 CBOW/SkipGram NS와
별도로 각 fused NS도 같은 모델의 Full Softmax와 비교한다. 평균의 95% 신뢰구간이
겹치지 않고 다음 vocabulary 지점에서도 NS 우위가 유지되는 첫 vocabulary 크기를
확정 교차점으로 출력한다. 단일 지점의 평균만
앞선 경우는 `first_observed_negative_sampling_win_vocab_size`에 남기되 교차점으로
확정하지 않는다. 현재 dense Adam 및 reporting loss까지 포함한 end-to-end update
교차점이며, objective 연산만의 이론적 교차점은 아니다.
추가로 `vsweap-cbow.png`와 `vsweap-skipgram.png`에 제목 없는 모델별 개별 그래프를
저장한다.
그래프의 측정 기준은 기본적으로 `--vsweap-timing window`이며, GPU CUDA event 기반의
steady update 평균을 사용하려면 `--vsweap-timing event`를 지정한다. event 모드에서는
host window 기반 95% 신뢰구간 band를 표시하지 않는다.
GPU 상태의 실행 순서 편향을 비교하려면 `--reverse-vocab-order`를 지정해 vocab size를
큰 값부터 측정할 수 있다. 실제 실행 순서는 결과 JSON의 `metadata.vocab_order`에 기록된다.
`--vocab-size`는 `--vsweap` 없이 사용할 수 없다.
