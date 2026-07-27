# DS2 원본 그림 재현

DS2 원본 책 코드가 실제로 표시하는 그림만 재현한다. 학습은 원본 저장소의
깨끗한 Git HEAD에서 수행하며, 그림은 저장된 CSV·NPZ 결과만 읽어 생성한다.

## 전체 실행

```bash
just exp ds2 run -o
```

고정 seed `1`로 그림이 있는 원본 시행을 모두 실행하고 결과를 저장한 다음,
PNG를 생성한다. 유효한 cache는 자동으로 건너뛰므로 중단 후 같은 명령으로
재개할 수 있다. MLflow 서버는 필요하지 않다.

결과 위치:

```text
exp/ds2/results/original/
├── data/eXX/<trial-id>/
│   ├── manifest.json
│   ├── *.csv
│   ├── *.npz
│   └── COMPLETE
└── image/
    └── *.png
```

## 학습 없이 다시 그리기

```bash
just exp ds2 analyze -o
```

저장된 CSV·NPZ만 사용한다. 원본 모델, Trainer, dataset을 import하거나 학습을
다시 실행하지 않는다. 필요한 cache가 없거나 artifact hash가 맞지 않으면
누락된 trial을 표시하고 실패한다.

## 일부 그림만 실행하거나 다시 그리기

`-e/--experiment`는 반복하거나 쉼표·범위로 지정할 수 있다.

```bash
# Toy CBOW만 학습하고 그림 생성
just exp ds2 run -o -e 01

# PTB CBOW와 Skip-gram 실행
just exp ds2 run -o -e 02

# Seq2seq와 attention 그림만 cache에서 다시 생성
just exp ds2 analyze -o -e 06-08

# 선택한 시행의 cache를 무시하고 다시 실행
just exp ds2 run -o -e 03 --force
```

## 생성 대상

| 실험 | 원본 그림 | 생성 수 |
| --- | --- | ---: |
| e01 | Toy CBOW loss | 1 |
| e02 | PTB CBOW loss, PTB Skip-gram loss | 2 |
| e03 | SimpleRnnlm perplexity | 1 |
| e04 | LSTM Rnnlm perplexity | 1 |
| e06 | Addition Seq2seq 조건별 accuracy | 4 |
| e07 | Date Seq2seq 조건별 accuracy | 3 |
| e08 | Attention alignment | 5 |
| 합계 |  | 17 |

e05 BetterRnnlm은 원본 학습 source에 `trainer.plot()`이나 `plt.show()`가 없으므로
실행·렌더 대상에서 제외한다. 프로젝트 확장 시행인 toy Skip-gram과 PTB
full-softmax 조건도 원본 그림 대상에 포함하지 않는다.

## 실행 backend

- e02는 원본의 `config.GPU`와 `to_gpu` 경로를 통해 CuPy를 사용한다.
- 그 외 그림 시행은 원본 NumPy 경로를 사용한다.
- CuPy 14에서는 원본 `common.np`의 `np.add.at` 대입이 허용되지 않으므로,
  adapter가 같은 scatter-add 의미를 유지하는 호환 모듈을 제공한다.
- 저장되는 checkpoint와 NPZ 배열은 모두 host NumPy 형식이다.

PTB Skip-gram 원본 학습은 수 시간이 걸릴 수 있다. 실행을 중단해도 완료된
trial cache는 유지되며, 전체 실행 명령을 다시 사용하면 미완료 trial부터
계속한다.

## 원본 모드 제약

원본 시행은 seed `1`과 source의 고정 hyperparameter를 사용한다. 다음 옵션은
원본 모드와 함께 사용할 수 없다.

- `--seed-set`
- seed `1` 이외의 `--seed`
- `--atomic-run`, `--exclude-atomic-run`
- `--set` YAML override
- `--device`
- `--seed-first`
