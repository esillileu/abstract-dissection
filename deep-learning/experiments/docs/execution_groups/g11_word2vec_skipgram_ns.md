# g11. word2vec_skipgram_ns

## 구조 서명

`ptb-skipgram-negative-sampling-v1`

## 묶음 기준

CBOW와 예측 그래프가 달라 분리

## 공통 실행 설정

PTB, window 5, embedding 100, batch 100, Adam .001, 10 epochs

## 원자 조건

| Atomic run ID | Override |
|---|---|
| `W2V-SG-NS` | Skip-gram; negative sampling=5 |

## 사용 실험

e09

## 실행 정책

- 확률적 조건은 seed `0..9`를 사용한다.
- 같은 구조 그룹 안에서 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- resolved config를 기준으로 condition key와 run key를 계산한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
