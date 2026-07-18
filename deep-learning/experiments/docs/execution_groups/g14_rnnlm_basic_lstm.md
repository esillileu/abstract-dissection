# g14. rnnlm_basic_lstm

## 구조 서명

`ptb-rnnlm-lstm100-v1`

## 묶음 기준

기본 LSTM 기준

## 공통 실행 설정

Embedding100-LSTM100-TimeAffine; SGD 20; BPTT35; 4 epochs

## 원자 조건

| Atomic run ID | Override |
|---|---|
| `LM-LSTM-C025` | LSTM; max_grad=.25 |

## 사용 실험

e11

## 실행 정책

- 확률적 조건은 seed `0..9`를 사용한다.
- 같은 구조 그룹 안에서 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- resolved config를 기준으로 condition key와 run key를 계산한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
