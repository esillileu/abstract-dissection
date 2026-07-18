# g04. batchnorm_off_mlp

## 구조 서명

`mnist-mlp-784-100x5-10-relu-no-bn-v1`

## 묶음 기준

BN 레이어 없음

## 공통 실행 설정

DS-MNIST-1000, SGD .01, batch 100, 20 epochs, float64

## 원자 조건

| Atomic run ID | Override |
|---|---|
| `BN-OFF-01` | BatchNorm=false; scale index 1; weight_init_scale=1 |
| `BN-OFF-02` | BatchNorm=false; scale index 2; weight_init_scale=0.541169527 |
| `BN-OFF-03` | BatchNorm=false; scale index 3; weight_init_scale=0.292864456 |
| `BN-OFF-04` | BatchNorm=false; scale index 4; weight_init_scale=0.158489319 |
| `BN-OFF-05` | BatchNorm=false; scale index 5; weight_init_scale=0.0857695899 |
| `BN-OFF-06` | BatchNorm=false; scale index 6; weight_init_scale=0.0464158883 |
| `BN-OFF-07` | BatchNorm=false; scale index 7; weight_init_scale=0.0251188643 |
| `BN-OFF-08` | BatchNorm=false; scale index 8; weight_init_scale=0.0135935639 |
| `BN-OFF-09` | BatchNorm=false; scale index 9; weight_init_scale=0.00735642254 |
| `BN-OFF-10` | BatchNorm=false; scale index 10; weight_init_scale=0.00398107171 |
| `BN-OFF-11` | BatchNorm=false; scale index 11; weight_init_scale=0.00215443469 |
| `BN-OFF-12` | BatchNorm=false; scale index 12; weight_init_scale=0.0011659144 |
| `BN-OFF-13` | BatchNorm=false; scale index 13; weight_init_scale=0.000630957344 |
| `BN-OFF-14` | BatchNorm=false; scale index 14; weight_init_scale=0.000341454887 |
| `BN-OFF-15` | BatchNorm=false; scale index 15; weight_init_scale=0.00018478498 |
| `BN-OFF-16` | BatchNorm=false; scale index 16; weight_init_scale=0.0001 |

## 사용 실험

e05

## 실행 정책

- 확률적 조건은 seed `0..9`를 사용한다.
- 같은 구조 그룹 안에서 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- resolved config를 기준으로 condition key와 run key를 계산한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
