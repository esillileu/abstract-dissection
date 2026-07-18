# g02. mnist_mlp_4hidden

## 구조 서명

`mnist-mlp-784-100x4-10-relu-v1`

## 묶음 기준

optimizer와 initializer만 변경

## 공통 실행 설정

DS-MNIST-FLAT, batch 128, 2,000 updates, sampling with replacement, float64

## 원자 조건

| Atomic run ID | Override |
|---|---|
| `MLP-SGD-HE` | SGD .01; He |
| `MLP-MOM-HE` | Momentum .01/.9; He |
| `MLP-ADAGRAD-HE` | AdaGrad .01; He |
| `MLP-ADAM-HE` | Adam .001; He |
| `MLP-SGD-XAVIER` | SGD .01; Xavier |
| `MLP-SGD-STD001` | SGD .01; Normal std=.01 |

## 사용 실험

e02, e04

## 실행 정책

- 확률적 조건은 seed `0..9`를 사용한다.
- 같은 구조 그룹 안에서 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- resolved config를 기준으로 condition key와 run key를 계산한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
