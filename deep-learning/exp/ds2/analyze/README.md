# DS2 분석 이관 목록

이 디렉터리는 실행 중 수집한 raw objective, evaluation, prediction, attention record에서
원본 series와 비교 통계를 만드는 책임을 정의한다. GT01-GT05의 pre-update curve는
`observations/source_objectives.csv`를 canonical 입력으로 사용한다.

## 공통 전처리

- schema-v1 config, seed, dataset, reproducibility artifact를 결합해 분석용 run manifest를 만든다.
- runtime metadata의 `dataset_checksum`, `split_checksum`을 condition 간에 검증한다.
- domain checkpoint CSV와 checkpoint manifest를 결합해 final/selected checkpoint index를 만든다.
- `source_objectives.csv`의 objective와 `unit_count`로 reducer를 적용하고 `plot_index`를
  append 순서 `0..N-1`로 생성한다.
- terminal 값이 `metrics/final.json`에만 있으면 long-form evaluation row로 정규화한다.
- prediction 문자열에서 token 통계를 재계산해 기록된 `token_correct`, `token_count`를 검증한다.
- seed 평균, 표준편차, CI, paired difference, final/best/AUC와 순위는 분석에서 계산한다.
- attention label, 축 반전, color range와 heatmap rendering은 분석·시각화에서 만든다.

## 실행 그룹별 작업

| 그룹 | 분석으로 이관한 항목 |
| --- | --- |
| `GT01` | `local_iteration % every_updates == 0` 규칙으로 원본 interval mean loss 재구성, append plot index, smoothing과 seed 집계 |
| `GT02` | GT01 reducer 적용, CBOW/Skip-gram 및 negative-sampling/full-softmax 조건 분리, word-vector checkpoint index와 조건 비교 |
| `GT03` | standard variant는 원본 zero-based interval PPL, custom variant는 epoch별 token-weighted mean NLL의 exp로 PPL 재구성; 두 loop의 축을 별도 유지 |
| `GT04` | interval token-weighted train PPL, terminal full-test PPL 정규화, train-test 비교와 seed CI |
| `GT05` | interval train PPL, epoch valid PPL, selected-checkpoint terminal test PPL 결합, lr history와 best-valid 선택 검증, recipe 순위 |
| `GT06` | epoch evaluation을 `plot_index=epoch-1` source series로 변환, epoch별 fixed prediction 변화와 exact/token accuracy 시각화 |
| `GT07` | GT06 처리에 모델별 비교를 추가하고 selected Attention checkpoint를 GO01 run과 연결 |
| `GO01` | seeded example ID 검증, source/target label 생성, input reversal에 맞춘 encoder 축 변환, attention heatmap과 prediction 표 생성 |

## Source Objective Reducer

- interval loss: trigger까지 objective의 산술 평균을 사용한다.
- interval PPL: `sum(objective * unit_count) / sum(unit_count)`의 exp를 사용한다.
- custom epoch PPL: epoch 전체에 같은 token-weighted reducer를 적용한다.
- 원본 Trainer의 interval trigger는 epoch-local `local_iteration`을 사용한다.
- `source_curves.csv`는 호환 projection으로만 사용하고 위 계산 결과와 교차 검증한다.

## 분석 입력 우선순위

1. GT01-GT05 curve 원재료는 `observations/source_objectives.csv`다.
2. GT06-GT07 accuracy는 `evaluations.csv`가 기준이고 source series는 여기서 파생한다.
3. prediction과 attention weight는 observation CSV가 기준이다.
4. run identity/config/checkpoint는 schema-v1 artifact와 domain CSV를 결합한다.

## 실행

완료된 MLflow seed trial만 조회하며 condition별로 현재 존재하는 모든 seed를 자동으로
집계한다. 공통 x축의 평균과 최솟값, 최댓값으로 원본 그래프를 복원한다.

```bash
python -m exp ds2 analyze --all --error-style band
python -m exp ds2 analyze -e 01-08 --error-style errorbar
```

`-e`는 `01`, `e01`, `01-08`, `01,03,06-08` 형식을 지원한다.

`--seed 1`은 MLflow의 실제 `seed/master=1`인 완료 run만 선택하며 출력 이름에
`_seed-1`을 붙인다.

- `--error-style band`: 평균선 주변 min–max 반투명 영역
- `--error-style errorbar`: 평균선 위 min–max error bar
- 완료 run이 없으면 `No completed runs` 빈 그래프와 값이 비어 있는 summary CSV를 만든다.
- 출력 기본 경로는 `exp/ds2/results/image/`이다.
- 확장 실험인 `GT05`와 `GT02`의 PTB full-softmax 조건은 제외한다.
- 원본 대상인 `e01`–`e04`, `e06`–`e08`을 지원한다.

각 실험의 reducer 결과 선택, 축과 원본 시각 형식은 `e01_toy_word2vec.py`부터
`e08_attention.py`까지의 개별 모듈이 소유한다. `common.py`에는 완료 run 조회와
source curve 로딩만 둔다.
