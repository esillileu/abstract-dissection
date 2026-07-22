# DS2 실행 catalog

DS2는 실행 그룹 `GT01`–`GT07`을 `e01`–`e07` YAML로 선언한다. 기존
`.legacy/experiments/deepscratch2`는 보존용 legacy이며 이 catalog의 입력이 아니다.

```bash
just exp ds2 plan -e 02 -seed 1
just exp ds2 run -e 02 -seed 1
```

`-e`는 실행 그룹에 대응하는 catalog ID이고, `-seed`/`--seed`는
`config/seeds.yaml`의 **시드 레지스트리 인덱스**다. 따라서 `-seed 1`은 두 번째
등록 시드를 선택한다. `e02=GT02`이고 네 개의 Word2Vec atomic trial을 모두
실행한다. 실제 실행 전에는 `just mlflow up`으로 MLflow를 시작한다.

| catalog ID | 실행 그룹 | YAML |
| --- | --- | --- |
| `e01` | `GT01` | `config/e01_toy_word2vec.yaml` |
| `e02` | `GT02` | `config/e02_ptb_word2vec.yaml` |
| `e03` | `GT03` | `config/e03_small_rnnlm.yaml` |
| `e04` | `GT04` | `config/e04_ptb_lstm_rnnlm.yaml` |
| `e05` | `GT05` | `config/e05_ptb_lm_recipes.yaml` |
| `e06` | `GT06` | `config/e06_addition_seq2seq.yaml` |
| `e07` | `GT07` | `config/e07_date_seq2seq.yaml` |

`GO01`은 GT07의 완료 checkpoint를 입력으로 받는 관찰 실행이라 학습 catalog에
포함하지 않는다.
