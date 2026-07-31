# DS2 실행 catalog

DS2는 실행 그룹 `GT01`–`GT07`을 `e01`–`e07` YAML로 선언한다. 기존
`.legacy/experiments/deepscratch2`는 보존용 legacy이며 이 catalog의 입력이 아니다.

```bash
just exp plan ds2 -e 02 --seed 1
just exp run ds2 -e 02 --seed 1
```

`-e`는 실행 그룹에 대응하는 catalog ID이고, `--seed`는
`config/seeds.yaml`에 등록된 **실제 master seed 값**이다. 기본 `research_v1`에서는
`1`부터 `10`까지를 사용한다. `e02=GT02`이고 여섯 개의 Word2Vec atomic trial을 모두
실행한다. 실제 실행 전에는 `just mlflow up`으로 MLflow를 시작한다.

특정 atomic run만 실행하려면 `-a`/`--atomic-run`, 특정 atomic run을 빼려면
`-x`/`--exclude-atomic-run`을 사용한다. 두 옵션은 함께 사용할 수 없으며, 반복하거나
쉼표로 여러 ID를 지정할 수 있다. ID는 선택한 `-e`/`--all` 범위 안에서 검증된다.

```bash
just exp plan ds2 -e 02 -a W2V-PTB-CBOW-NS --seed 1
just exp run ds2 -e 02 -a W2V-PTB-CBOW-NS,W2V-PTB-SKIPGRAM-NS --seed 1
just exp run ds2 -e 02 -x W2V-PTB-CBOW-FULL --seed 1
```

기본 실행 순서는 atomic run 우선이다. 선택한 모든 atomic run을 같은 seed끼리 먼저
실행하려면 `--order seed-first`를 추가한다. 여러 experiment를 선택해도 전체 plan에
적용되며, `--seed`에 지정한 값의 순서를 따른다.

```bash
just exp plan ds2 -e 01-02 --seed 1-3 --order seed-first
just exp run ds2 -e 01-02 --seed 1-3 --order seed-first
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

`GO01`은 GT07의 완료 checkpoint를 입력으로 받는 관찰 실행이다.
