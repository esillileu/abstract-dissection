# DS2 실행 catalog

DS2는 실행 그룹 `GT01`–`GT07`, `GT09`–`GT10`과 성능 프로파일 그룹 `PF01`–`PF02`를
`e01`–`e12` YAML로 선언한다. `e10/PF01`은 `e02`에서 파생된 PTB Word2Vec
update profile이고, `e11/PF02`는 synthetic vocabulary-size scaling study다.
과거 실행은 MLflow의 historical namespace에 보존되지만 operational 명령에서는
canonical `deepscratch.ds2` namespace만 조회한다.

```bash
just exp plan deepscratch ds2 -e 02 --seed 1
just exp run deepscratch ds2 -e 02 --seed 1
```

`-e`는 실행 그룹에 대응하는 catalog ID이고, `--seed`는
`config/seeds.yaml`에 등록된 **실제 master seed 값**이다. 기본 `research_v1`에서는
`1`부터 `10`까지를 사용한다. `e02=GT02`이고 여섯 개의 Word2Vec atomic trial을 모두
실행한다. 실제 실행 전에는 `just mlflow up`으로 MLflow를 시작한다.

특정 atomic run만 실행하려면 `-a`/`--atomic-run`, 특정 atomic run을 빼려면
`-x`/`--exclude-atomic-run`을 사용한다. 두 옵션은 함께 사용할 수 없으며, 반복하거나
쉼표로 여러 ID를 지정할 수 있다. ID는 선택한 `-e`/`--all` 범위 안에서 검증된다.

```bash
just exp plan deepscratch ds2 -e 02 -a W2V-PTB-CBOW-NS --seed 1
just exp run deepscratch ds2 -e 02 -a W2V-PTB-CBOW-NS,W2V-PTB-SKIPGRAM-NS --seed 1
just exp run deepscratch ds2 -e 02 -x W2V-PTB-CBOW-FULL --seed 1
```

`e10/PF01` 프로파일은 seed trial이 아닌 single-run atomic profile이다. 기존
Planner/Runner와 durable MLflow lifecycle을 그대로 사용하며, 결과 분석은
`run.type=profile`만 선택한다. 프로파일은 비용과 측정 protocol이 일반 학습과
다르므로 `--all`에서는 제외되고 `-e 10` 또는 `-e 11`로 명시해야 한다.

```bash
just exp plan deepscratch ds2 -e 10
just exp run deepscratch ds2 -e 10 -a PF-W2V-CBOW-IMPLEMENTED-NS
just exp check deepscratch ds2 -e 10
just exp analyze deepscratch ds2 -e 10
```

`e11/PF02`는 vocabulary의 내용이 아니라 독립변수 `V`를 변화시키므로 공식
명칭과 코드 식별자를 모두 vocabulary-size scaling으로 사용한다.

```bash
just exp plan deepscratch ds2 -e 11
just exp run deepscratch ds2 -e 11
just exp check deepscratch ds2 -e 11
just exp analyze deepscratch ds2 -e 11
```

프로파일 분석은 기본적으로 `cuda:0`의 `window` timing 결과를 고른다. 다른
측정 좌표는 명시적으로 선택한다.

```bash
just exp analyze deepscratch ds2 -e 10 \
  --profile-device cpu \
  --profile-timing-source window
```

e10은 update 비교와 module breakdown PNG/CSV를 함께 만들고, e11은
vocabulary-size scaling PNG/CSV/Markdown을 만든다. 파일명에는 선택 장치가
`_cuda0` 또는 `_cpu` suffix로 포함된다.

기본 실행 순서는 atomic run 우선이다. 선택한 모든 atomic run을 같은 seed끼리 먼저
실행하려면 `--order seed-first`를 추가한다. 여러 experiment를 선택해도 전체 plan에
적용되며, `--seed`에 지정한 값의 순서를 따른다.

```bash
just exp plan deepscratch ds2 -e 01-02 --seed 1-3 --order seed-first
just exp run deepscratch ds2 -e 01-02 --seed 1-3 --order seed-first
```

| catalog ID | 실행 그룹 | YAML |
| --- | --- | --- |
| `e01` | `GT01` | `config/e01_toy_word2vec.yaml` |
| `e02` | `GT02` | `config/e02_ptb_word2vec.yaml` |
| `e03` | `GT03` | `config/e03_small_rnnlm.yaml` |
| `e04` | `GT04` | `config/e04_ptb_lstm_rnnlm.yaml` |
| `e05` | `GT05` | `config/e05_ptb_lm_recipes.yaml` |
| `e06` | `GT06` | `config/e06_addition_seq2seq.yaml` |
| `e07` | `GT07` | `config/e07_date_seq2seq.yaml` |
| `e08` | `GO01` | `config/e08_attention_alignment.yaml` |
| `e09` | `GT09` | `config/e09_addition_seq2seq_150.yaml` |
| `e10` | `PF01` | `config/implemented/e10_ptb_word2vec_profile.yaml` |
| `e11` | `PF02` | `config/implemented/e11_word2vec_vocabulary_size_scaling.yaml` |
| `e12` | `GT10` | `config/implemented/e12_count_based_embeddings.yaml` |

`e12/GT10`은 교재 2장의 PTB window 2 조건으로 PPMI 행, full SVD 100차원,
randomized SVD 100차원을 비교한다. 공기행렬·PPMI·분해 결과를
`statistical_matrices.npz`에, 분석용 벡터를 final checkpoint에, 단계별 wall time을
`timing.json`에 저장한다. full SVD는 PTB 전체 dense 행렬에서 계산량과 메모리 사용량이
크므로 필요한 조건만 atomic run으로 선택할 수 있다.

```bash
just exp run deepscratch ds2 -e 12 --seed 1
just exp analyze deepscratch ds2 -e 12 --seed 1 --summary
```

`GO01`은 GT07의 완료 checkpoint를 입력으로 받는 관찰 실행이다.
