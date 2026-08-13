# DS2 분석 이관 목록

이 디렉터리는 실행 중 수집한 raw objective, evaluation, prediction, attention record에서
원본 series와 비교 통계를 만드는 책임을 정의한다. GT01-GT05의 pre-update curve는
`observations/source_objectives.csv`를 canonical 입력으로 사용한다.

## 공통 전처리

- schema-v1 config, seed, dataset, reproducibility artifact를 결합해 분석용 run manifest를 만든다.
- runtime metadata의 `dataset_checksum`, `split_checksum`을 condition 간에 검증한다.
- domain checkpoint CSV와 checkpoint manifest를 결합해 final/selected checkpoint index를 만든다.
- `source_objectives.csv`의 objective/book_objective와 `unit_count`로 reducer를
  적용하고 `plot_index`를 append 순서 `0..N-1`로 생성한다. GT01 원본형
  그래프는 book_objective를 사용하고 prediction-term mean은 진단값으로 유지한다.
- terminal 값이 `metrics/final.json`에만 있으면 long-form evaluation row로 정규화한다.
- prediction 문자열에서 token 통계를 재계산해 기록된 `token_correct`, `token_count`를 검증한다.
- seed 평균, 표준편차, CI, paired difference, final/best/AUC와 순위는 분석에서 계산한다.
- attention label, 축 반전, color range와 heatmap rendering은 분석·시각화에서 만든다.

## 실행 그룹별 작업

| 그룹 | 분석으로 이관한 항목 |
| --- | --- |
| `GT01` | `local_iteration % every_updates == 0` 규칙으로 원본 interval mean loss 재구성, append plot index, toy CBOW/Skip-gram 비교와 seed 집계 |
| `GT02` | final `W_in` checkpoint에 교재의 고정 유사어·유추 질의를 적용하고 seed별 top-5 후보, 기대 정답 순위, hit@5를 출력 |
| `GT03` | 원본 zero-based interval train PPL과 seed CI |
| `GT04` | interval token-weighted train PPL, terminal full-test PPL 정규화, train-test 비교와 seed CI |
| `GT05` | epoch valid PPL을 broken y-axis로 비교해 vanilla RNNLM과 나머지 두 recipe의 큰 scale 차이를 분리 |
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
python -m exp analyze deepscratch ds2 --all
python -m exp analyze deepscratch ds2 -e 01-08
```

`-e`는 `01`, `e01`, `01-08`, `01,03,06-08` 형식을 지원한다.

`--seed 1`은 MLflow의 실제 `seed/master=1`인 완료 run만 선택하며 출력 이름에
`_seed-1`을 붙인다.

`--legacy`는 `protocol.version=legacy`인 이전 완료 run을 선택한다. 최신 protocol
분석 결과를 덮어쓰지 않도록 출력 이름에 `_legacy`를 붙이며, `--seed`와 함께 쓸 수
있다. 이전 철자와의 호환을 위해 `--lagacy`도 같은 옵션으로 인식한다.

```bash
python -m exp analyze deepscratch ds2 -e 03-07 --variant all
python -m exp analyze deepscratch ds2 -e 03 --variant all --seed 1
```

각 분석은 출력 옆의 숨김 `*.analysis-cache.json` manifest에 조회 조건과 선택된 run
ID를 기록한다. 다음 실행에서 run ID와 출력 파일이 모두 같으면 아티팩트를 다시 받거나
렌더링하지 않고 기존 결과 경로를 출력한다. 그래프와 `--summary`는 서로 다른 manifest를
사용한다.

## 최종 metric과 학습 시간 요약

`-e 01 --summary`부터 `-e 07 --summary`는 그림 없이 각 atomic run의 최종
metric과 train wall time을 `평균 ± 표본표준편차, [최소, 최대]`로 터미널과
CSV에 출력한다. `-s`는 `--summary`의 단축 옵션이며 CSV 이름은
`e01_summary.csv` 형식이다. 각 seed run의 `model/parameter_manifest.json`에서
합산한 모델별 `parameter_count`도 함께 출력하며 seed 간 값이 같은지 검증한다.
E02는 `profiles/profiling_summary.json`의
`runtime.train_synchronized.mean_ms`를 우선 사용한다. corrected run이 하나라도
있으면 과거 비동기 `timing_windows.csv` 값과 섞지 않고 corrected run만 집계한다.
그 밖의 그룹은 synchronized metric이 없을 때 기존 train wall window 합계를 사용한다.

- E01–E02: 마지막 source-curve `book_loss`, 원본 스케일 소수점 세 자리.
  E02의 값은 objective 간 품질 비교가 아니라 재현 조건 확인용이다.
- E03: 마지막 train perplexity, 원본 스케일 소수점 두 자리
- E04–E05: terminal test perplexity, 원본 스케일 소수점 두 자리
- E06–E07: 마지막 test exact-match accuracy, 백분율 소수점 두 자리
- 학습 시간: 초 단위 소수점 한 자리

E08은 checkpoint를 읽어 attention alignment를 관찰하는 실행이며 자체 학습이나
최종 성능 metric이 없으므로 이 요약 ID를 제공하지 않는다.

원본 코드는 이제 `ds2_original` 정식 도메인에서 seed별로 분석한다.

```bash
python -m exp analyze deepscratch ds2 -e 01,03-04,06-08 --variant original
```

이 경로는 MLflow 재현 run을 조회하지 않고
기존 고정-seed 캐시의 canonical 격리 위치는 `.legacy/experiments/ds2_original/fixed_seed/`이며 새 통계에서 제외된다. source-tree의 기존 payload는 감사 전 read-only fallback으로만 읽는다.
원본 시행이 없는 확장 E05와 관찰 전용 E08은 제외한다. 단일 원본 시행은
`seed_runs=1`, 표준편차 `0`으로 기록한다. 기존 원본 캐시에 학습 시간이나
parameter count가 없으면 해당 CSV 행은 빈 값으로 남긴다. 계측 schema-v2
runner로 재실행하면 synchronized training wall time과 공유 텐서를 중복 제거한
parameter count를 `timing.json`, `parameter_manifest.json`에서 읽는다.

- `--error-style band`: 평균선 주변 ±1 표본 표준편차 반투명 영역
- `--error-style errorbar`: 평균선 위 min–max error bar
- 완료 run이 없으면 빈 그래프 또는 값이 비어 있는 분석 CSV를 만든다.
- 출력 기본 경로는 `.artifacts/experiments/deepscratch/ds2/<experiment>/implemented/analysis/`이다.
- `GT01`은 같은 toy/full-softmax 조건의 CBOW와 Skip-gram을 함께 표시한다.
- `GT02`는 그래프를 만들지 않는다. 네 조건의 final `W_in`에 대해 교재의
  `you`, `year`, `car`, `toyota` 유사어 top-5와 네 유추 문제의 top-5,
  기대 정답 순위 및 hit@5를 터미널과 `e02_word_vectors.txt/.csv`에 출력한다.
- `GT05`는 validation perplexity만 표시한다. 아래의 확대 축은 LSTM/BetterRNNLM
  최댓값 바로 위에서 끊고, 작은 위쪽 축에는 훨씬 큰 vanilla RNNLM 범위를 표시한다.
- `e01`–`e09`를 지원한다.

각 실험의 reducer 결과 선택, 축과 원본 시각 형식은 `e01_toy_word2vec.py`부터
`e09_addition_seq2seq_150.py`까지의 개별 모듈이 소유한다. `common.py`에는 완료 run 조회와
source curve 로딩만 둔다.
