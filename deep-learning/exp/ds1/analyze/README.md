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
| `GT06` | epoch-first subset accuracy와 terminal full-test accuracy 결합, learning curve, final checkpoint lookup, seed 집계 |
| `GT07` | GT06과 같은 축·checkpoint 처리, SimpleCNN 대비 DeepCNN 비교, runtime/accuracy trade-off |
| `GT08` | epoch-end full-test curve, spatial/permuted paired comparison, NN/CNN 구조별 차이와 seed CI |
| `GO01` | 저장된 `0..29` trajectory update를 표시용 `1..30`으로 변환, optimizer별 경로·등고선 시각화 |
| `GO02` | histogram count를 density/ratio로 변환, bin·layer 정렬, summary 검증과 activation 분포 시각화 |

## 분석 입력 우선순위

1. 수치 history는 domain CSV를 기준으로 한다.
2. run identity와 config는 schema-v1 config artifact를 기준으로 한다.
3. final checkpoint 정보는 checkpoint manifest를 기준으로 하고 domain CSV와 교차 검증한다.
4. `metrics/final.json`은 편의 summary이며 raw history를 대체하지 않는다.
