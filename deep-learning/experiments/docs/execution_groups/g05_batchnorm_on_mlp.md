# g05. batchnorm_on_mlp

## 구조 서명

`mnist-mlp-784-100x5-10-bn-relu-v1`

## 묶음 기준

BN 레이어 추가로 g04와 구조 분리

## 공통 실행 설정

DS-MNIST-1000, Affine-BN-ReLU x5, SGD .01, batch 100, 20 epochs

## 원자 조건

| Atomic run ID | Override |
|---|---|
| `BN-ON-01` | BatchNorm=true; scale index 1; weight_init_scale=1 |
| `BN-ON-02` | BatchNorm=true; scale index 2; weight_init_scale=0.541169527 |
| `BN-ON-03` | BatchNorm=true; scale index 3; weight_init_scale=0.292864456 |
| `BN-ON-04` | BatchNorm=true; scale index 4; weight_init_scale=0.158489319 |
| `BN-ON-05` | BatchNorm=true; scale index 5; weight_init_scale=0.0857695899 |
| `BN-ON-06` | BatchNorm=true; scale index 6; weight_init_scale=0.0464158883 |
| `BN-ON-07` | BatchNorm=true; scale index 7; weight_init_scale=0.0251188643 |
| `BN-ON-08` | BatchNorm=true; scale index 8; weight_init_scale=0.0135935639 |
| `BN-ON-09` | BatchNorm=true; scale index 9; weight_init_scale=0.00735642254 |
| `BN-ON-10` | BatchNorm=true; scale index 10; weight_init_scale=0.00398107171 |
| `BN-ON-11` | BatchNorm=true; scale index 11; weight_init_scale=0.00215443469 |
| `BN-ON-12` | BatchNorm=true; scale index 12; weight_init_scale=0.0011659144 |
| `BN-ON-13` | BatchNorm=true; scale index 13; weight_init_scale=0.000630957344 |
| `BN-ON-14` | BatchNorm=true; scale index 14; weight_init_scale=0.000341454887 |
| `BN-ON-15` | BatchNorm=true; scale index 15; weight_init_scale=0.00018478498 |
| `BN-ON-16` | BatchNorm=true; scale index 16; weight_init_scale=0.0001 |

## 사용 실험

e05

## 실행 정책

- 확률적 조건은 seed `0..9`를 사용한다.
- 같은 구조 그룹 안에서 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- resolved config를 기준으로 condition key와 run key를 계산한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
