# g06. regularization_no_dropout

<!-- Domain: deepscratch1 -->

## 구조 서명

`mnist-mlp-784-100x6-10-relu-no-dropout-v1`

## 묶음 기준

L2는 loss/gradient 항만 변경

## 공통 실행 설정

DS-MNIST-300, He, SGD .01, batch 100, 301 epochs

## 원자 조건

| Atomic run ID | Override |
|---|---|
| `REG-BASE` | lambda=0 |
| `REG-WD-1E4` | lambda=1e-4 |
| `REG-WD-1E3` | lambda=1e-3 |
| `REG-WD-1E2` | lambda=1e-2 |
| `REG-WD-1E1` | lambda=1e-1 |

## 사용 실험

e06, e07 baseline

## 실행 정책

- 확률적 조건은 seed `0..9`를 사용한다.
- 같은 구조 그룹 안에서 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- resolved config를 기준으로 condition key와 run key를 계산한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
