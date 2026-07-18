# e10. Full softmax-Negative sampling

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e10` |
| 데이터·태스크 | PTB CBOW 중심어 예측 |
| 실험 목적 | 출력 objective만 바꿨을 때 계산시간, 메모리, 임베딩 품질 trade-off를 확인한다. |
| 사전 가설 | negative sampling은 full softmax보다 빠르고 메모리 사용이 작으며 품질 감소는 제한적이다. |
| 독립변인 | full vocabulary softmax, negative sampling |
| 고정변수 | CBOW, PTB, window 5, embedding 100, batch 100, Adam .001, 10 epochs |
| 종속변인·관찰값 | normalized loss, 1회 update 시간, 전체 학습시간, throughput, peak memory, output 연산량, 임베딩 평가 |

## 2. 분석 계획

시스템 비용을 주 분석으로 두고 동일 seed의 품질 감소를 비열등성 관점에서 본다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

negative sampling의 1회 업데이트 시간이 더 짧고, 학습 완료 후 평가 정확도 감소가 1% 이하이다.

## 4. 조회할 원자 실행

`W2V-CBOW-FULL`, `W2V-CBOW-NS`

## 5. 해석 제한

full softmax와 negative sampling의 목적함수 값 자체는 동일 척도가 아닐 수 있다.
