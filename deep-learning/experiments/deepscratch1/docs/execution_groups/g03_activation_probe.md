# g03. activation_probe

<!-- Domain: deepscratch1 -->

## 구조 서명

`activation-probe-100x5-v1`

## 묶음 기준

activation 함수와 scale만 변경

## 공통 실행 설정

DS-SYNTH-ACT, width 100, hidden depth 5, no training, float64

## 원자 조건

| Atomic run ID | Override |
|---|---|
| `ACT-SIG-STD1` | sigmoid; std=1 |
| `ACT-SIG-STD001` | sigmoid; std=.01 |
| `ACT-SIG-XAVIER` | sigmoid; Xavier |
| `ACT-SIG-HE` | sigmoid; He |
| `ACT-TANH-STD1` | tanh; std=1 |
| `ACT-TANH-STD001` | tanh; std=.01 |
| `ACT-TANH-XAVIER` | tanh; Xavier |
| `ACT-TANH-HE` | tanh; He |
| `ACT-RELU-STD1` | relu; std=1 |
| `ACT-RELU-STD001` | relu; std=.01 |
| `ACT-RELU-XAVIER` | relu; Xavier |
| `ACT-RELU-HE` | relu; He |

## 사용 실험

e03

## 실행 정책

- 확률적 조건은 seed `0..9`를 사용한다.
- 같은 구조 그룹 안에서 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- resolved config를 기준으로 condition key와 run key를 계산한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
