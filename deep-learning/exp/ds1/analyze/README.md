# DS1 분석 이관 목록

이 디렉터리는 실행 중 수집하지 않고 결과 전처리·분석에서 계산할 항목을 정의한다.
분석 코드는 `updates.csv`, `evaluations.csv`, `timing_windows.csv`, observation CSV와
상위 schema-v1 artifact를 입력으로 사용한다. 실행 artifact를 수정하지 않고 파생 결과를
별도 출력 디렉터리에 쓴다.

## 공통 전처리

- `config/resolved.json`, `config/seed.json`, `data/dataset_manifest.json`,
  `reproducibility/runtime.json`을 결합해 분석용 run manifest를 만든다.
- `checkpoints.csv`와 `checkpoints/checkpoint_manifest.json`을 결합해 final, periodic,
  selected checkpoint index를 만든다.
- CSV record에서 MLflow metric name/step을 재구성하고 누락·중복 step을 검증한다.
- seed trial을 condition 단위로 묶고 평균, 표준편차, CI와 paired difference를 계산한다.
- smoothing, final/best, AUC, threshold 도달 update, 순위는 모두 분석 산출물로 만든다.
- timing ns를 단위 변환하고 update당 시간, probe overhead, 처리량을 계산한다.

## 실행 그룹별 작업

| 그룹 | 분석으로 이관한 항목 |
| --- | --- |
| `GT01` | raw update loss smoothing, optimizer별 learning curve 정렬, final/best/AUC, seed 집계와 optimizer 순위 |
| `GT02` | initializer별 raw loss smoothing, 발산·NaN 구간 표시, threshold 도달 update, seed 집계 |
| `GT03` | update축 accuracy 정렬, train-test gap, weight-decay paired difference, final/best/AUC와 seed CI |
| `GT04` | `epoch_first_update` 평가를 epoch plot index로 변환, dropout on/off·ratio 비교, generalization gap과 seed CI |
| `GT05` | atomic run ID에서 scale와 BN on/off 조건 복원, scale축 reshape, paired BN difference, epoch curve와 순위 |
| `GT06` | 원본 SimpleConvNet train/test learning curve 형식으로 seed 평균과 ±1 표본 표준편차 범위 표시, terminal full-test accuracy와 final checkpoint lookup |
| `GT07` | GT06과 같은 축·checkpoint 처리, SimpleCNN 대비 DeepCNN 비교, runtime/accuracy trade-off |
| `GT08` | epoch-end full-test curve, spatial/permuted paired comparison, NN/CNN 구조별 차이와 seed CI, MNIST 원본/고정 pixel-permutation 예시 |
| `GO01` | 저장된 `0..29` trajectory update를 표시용 `1..30`으로 변환, optimizer별 경로·등고선 시각화 |
| `GO02` | histogram count를 density/ratio로 변환, bin·layer 정렬, summary 검증과 activation 분포 시각화 |
| `E11` | GT06 SimpleCNN과 GT08 CNN identity/permuted의 seed index 0 final checkpoint 첫 합성곱 필터를 공통 색상 범위의 3개 패널로 비교 |

## 분석 입력 우선순위

1. 수치 history는 domain CSV를 기준으로 한다.
2. run identity와 config는 schema-v1 config artifact를 기준으로 한다.
3. final checkpoint 정보는 checkpoint manifest를 기준으로 하고 domain CSV와 교차 검증한다.
4. `metrics/final.json`은 편의 summary이며 raw history를 대체하지 않는다.

## 실행

완료된 MLflow seed trial만 조회하며 같은 condition/seed가 여러 번 실행됐으면 가장
최근 완료 run을 사용한다. 모든 seed에 공통으로 존재하는 x축만 정렬해 평균과 최솟값,
최댓값을 계산한다.

```bash
python -m exp ds1 analyze --all --error-style band
python -m exp ds1 analyze -e 01-07 --error-style errorbar
```

`-e`는 `01`, `e01`, `01-07`, `01,03,05-07` 형식을 지원한다.

특정 완료 seed만 그리려면 MLflow의 실제 `seed/master` 값을 지정한다.

```bash
python -m exp ds1 analyze -e 06 --seed 1
```

이때 출력 이름에는 `_seed-1`이 붙어 전체-seed 집계 그래프를 덮어쓰지 않는다.

- `--error-style band`: 평균선 주변 ±1 표본 표준편차 반투명 영역
- `--error-style errorbar`: 평균선 위 min–max error bar
- 완료 run이 없으면 `No completed runs` 빈 그래프와 값이 비어 있는 summary CSV를 만든다.
- 출력 기본 경로는 `exp/ds1/results/image/`이다.
- 명시적으로 포함한 spatial-layout 비교 `GT08`은 `e08` 분석으로 제공한다.
- `e01`–`e11`을 지원한다.

각 실험의 artifact 선택, 축 변환, subplot 구성과 원본 시각 형식은
`e01_optimizer.py`부터 `e11_cnn_filters.py`까지의 개별 모듈이 소유한다.
`common.py`에는 DS1 CSV를 loss/accuracy curve로 읽는 공통 연산만 둔다.

`e06`은 원본 `train_convnet.py`처럼 단일 축에 SimpleConvNet의 train·test 곡선을
그리고, 10회 seed 평균 주변의 ±1 표본 표준편차 범위를 표시한다. 원본과 같은 `6.4 × 4.8`
canvas, epoch 축, `0–1` accuracy 범위, marker와 범례 위치를 사용하되 저장소 테마를
적용한다. 출력은 `e06_{band,errorbar}.png`이다.

`e07`은 SimpleCNN/DeepCNN의 train·test 네 곡선을 비교하고 y축 `0.25–0.9`를
생략한다. 출력은 `e07_{band,errorbar}.png`이다.

`-e 06 --summary`와 `-e 07 --summary`는 원본 실행의 마지막 train/test 정확도와
10회 seed 실행의 마지막 정확도 `평균 ± 표본 표준편차`를 나란히 출력한다. 양쪽 모두
학습 곡선과 같은 first-1000 평가를 사용한다. 원본은
`results/original/data/e{06,07}/.../metrics.csv`, 10회 실행은 각각
`mnist-train-first-1000`, `mnist-test-first-1000` 평가를 기준으로 한다.
정확도는 백분율로 항상 소수점 둘째 자리까지 표시한다. 학습 시간은 원본 캐시에
실측값이 없으므로 `results/original/cupy_estimate.json`의
`projected_update_time_s`를 `original projected`로 명시하고, 재현 실행은
`timing_windows.csv`의 평가 시간을 제외한 순수 학습 wall time 합계를
10회 `평균 ± 표본 표준편차`로 표시한다.

`-e 08 --summary`는 각 조건의 seed별 마지막 `mnist-test-full` 정확도와
`timing_windows.csv`의 train wall time 합계를 집계한다. `-s`는 `--summary`의
단축 옵션이다. E08은
`NN-MATCHED`, `NN-MATCHED-PERMUTED`, `CNN-SIMPLE-SPATIAL`,
`CNN-SIMPLE-SPATIAL-PERMUTED`를 각각 출력한다. 출력 CSV의 `mean`,
`standard_deviation`, `minimum`, `maximum`이 각각 seed 평균, 표본
표준편차, 최솟값, 최댓값이며 학습 시간 단위는 초다. 그림은 만들지 않고
`평균 ± 표준편차, [최소, 최대]`를 터미널에 바로 출력한다. 정확도는 백분율로
항상 소수점 둘째 자리까지, 학습 시간은 초 단위로 항상 소수점 첫째 자리까지
표시하며 CSV도 같은 단위와 자릿수를 사용한다.

summary CSV 이름은 `e06_summary.csv`, `e07_summary.csv`,
`e08_summary.csv` 형식이다.

원본 코드의 고정 seed 실행 캐시만 요약하려면 `--original`을 함께 사용한다.

```bash
python -m exp ds1 analyze --original -e 01-07 -s
```

이 경로는 MLflow 재현 run을 조회하지 않고
`exp/ds1/results/original/data/`의 `metrics.csv`와 `manifest.json`만 읽는다.
원본 최종 성능 metric이 있는 E01–E07을 지원하며 관찰 전용 E09–E10은 제외한다.
단일 원본 시행은 `seed_runs=1`, 표준편차 `0`으로 기록한다. 기존 원본 캐시에
학습 시간이나 parameter count가 없으면 해당 CSV 행은 빈 값으로 남긴다. 계측
schema-v2 runner로 재실행하면 synchronized training wall time과 공유 텐서를
중복 제거한 parameter count를 `timing.json`, `parameter_manifest.json`에서 읽는다.

`e08`은 `update/eval_{train,test}/accuracy`를 사용한다. 왼쪽은 NN identity/permuted,
오른쪽은 CNN identity/permuted이며 test는 실선, train은 점선이다. 두 패널 모두 같은
broken y축을 사용한다.

`e11`은 원본 교재의 `filter_show`를 final checkpoint 분석으로 확장한다. seed 간에는
필터 순서가 대응하지 않으므로 평균하지 않고 완료 seed별 이미지를 만든다. 첫 합성곱층은
교재처럼 출력 필터를 격자로 배치한다. DeepCNN의 후속 합성곱층은 각 출력 필터 안에 모든
입력 채널 커널을 작은 격자로 다시 배치해 채널 축을 평균하거나 버리지 않는다. 산출물은
`e11_{band,errorbar}/` 아래에 저장하며 CSV에는 checkpoint와 층별 shape·가중치 요약을
기록한다.
