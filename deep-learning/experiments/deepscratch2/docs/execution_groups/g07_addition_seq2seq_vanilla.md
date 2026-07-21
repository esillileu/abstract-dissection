# g07. addition_seq2seq_vanilla

## 구조 서명

`addition-vanilla-seq2seq-v1`

## 묶음 기준

입력 순서만 변경

## 공통 실행 설정

DS-SEQ-ADD, embedding16, hidden128, batch128, 25 epochs, Adam .001

## 원자 조건

| Atomic run ID | Override |
|---|---|
| `SEQA-VAN-FWD` | reverse=false |
| `SEQA-VAN-REV` | reverse=true |

## 사용 실험

e04

## 실행 정책

- 확률적 조건은 seed `0..9`를 사용한다.
- 같은 구조 그룹 안에서 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- resolved config를 기준으로 condition key와 run key를 계산한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
