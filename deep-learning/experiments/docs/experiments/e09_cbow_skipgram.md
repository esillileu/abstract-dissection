# e09. CBOW-Skip-gram

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e09` |
| 데이터·태스크 | PTB 단어 임베딩 학습 |
| 실험 목적 | 구조에 따른 학습 비용, normalized loss, 임베딩 특성의 차이를 비교한다. |
| 사전 가설 | 두 구조 모두 학습되지만 Skip-gram은 target당 예측 항이 많아 업데이트 비용이 더 크다. |
| 독립변인 | CBOW, Skip-gram |
| 고정변수 | PTB, window 5, embedding 100, negative sample 5, batch 100, Adam .001, 10 epochs |
| 종속변인·관찰값 | prediction term당 normalized loss, corpus 규모별 평가 정확도, 업데이트 시간, throughput, memory, nearest neighbor, analogy |

## 2. 분석 계획

raw loss 대신 예측 항당 loss를 사용하고, corpus 규모별 품질-비용 곡선을 비교한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

두 모델 모두 loss가 감소하며, corpus 규모별 품질 곡선과 업데이트 시간 분포에 유의미한 차이가 나타난다.

## 4. 조회할 원자 실행

`W2V-CBOW-NS`, `W2V-SG-NS`

## 5. 해석 제한

CBOW와 Skip-gram의 raw summed loss를 직접 비교하지 않는다.
