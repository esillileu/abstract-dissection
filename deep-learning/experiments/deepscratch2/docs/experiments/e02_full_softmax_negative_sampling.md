# e02. Full softmax-Negative sampling

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e02` |
| 데이터·태스크 | PTB 중심어/주변 단어 예측 |
| 실험 목적 | CBOW와 Skip-gram 각각에서 출력 objective를 바꿨을 때 계산시간, 메모리, 임베딩 품질 trade-off를 확인한다. |
| 사전 가설 | 두 구조 모두 negative sampling은 full softmax보다 빠르고 메모리 사용이 작으며 품질 감소는 제한적이다. |
| 독립변인 | architecture `{CBOW, Skip-gram}` × objective `{full vocabulary softmax, negative sampling}` |
| 고정변수 | PTB, window 5, embedding 100, batch 100, Adam .001, 10 epochs, summed prediction-term loss, book Word2Vec trainer |
| 종속변인·관찰값 | 구조별 normalized loss 추이, 1회 update 시간, 전체 학습시간, throughput, peak memory, output 연산량, 임베딩 평가 |

## 2. 분석 계획

시스템 비용을 주 분석으로 두고, 같은 architecture·seed 안에서 full softmax와 negative sampling을 짝지어 품질 감소를 비열등성 관점에서 본다. 구조 간 비용 차이는 별도 보조 분석으로 보고 objective 효과와 혼동하지 않는다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

각 architecture에서 negative sampling의 1회 업데이트 시간이 더 짧고, 학습 완료 후 평가 정확도 감소가 1% 이하이다.

## 4. 조회할 원자 실행

`W2V-CBOW-FULL`, `W2V-CBOW-NS`, `W2V-SG-FULL`, `W2V-SG-NS`

## 5. 해석 제한

full softmax와 negative sampling의 목적함수 값 자체는 동일 척도가 아닐 수 있다. 또한 Skip-gram은 중심어 하나에 여러 주변 단어를 예측하므로, architecture 간 loss 절대값 비교는 하지 않는다.
