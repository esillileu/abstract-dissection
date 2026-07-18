# g07. regularization_dropout

## 구조 서명

`mnist-mlp-784-100x6-10-relu-dropout-v1`

## 묶음 기준

Dropout 레이어 추가로 g06과 구조 분리

## 공통 실행 설정

DS-MNIST-300, dropout after each hidden ReLU, He, SGD .01, 301 epochs

## 원자 조건

| Atomic run ID | Override |
|---|---|
| `REG-DO-01` | dropout=.1 |
| `REG-DO-02` | dropout=.2 |
| `REG-DO-03` | dropout=.3 |
| `REG-DO-05` | dropout=.5 |

## 사용 실험

e07

## 실행 정책

- 확률적 조건은 seed `0..9`를 사용한다.
- 같은 구조 그룹 안에서 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- resolved config를 기준으로 condition key와 run key를 계산한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
