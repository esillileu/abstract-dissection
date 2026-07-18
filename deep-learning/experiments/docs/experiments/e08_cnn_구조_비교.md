# e08. CNN 구조 비교

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e08` |
| 데이터·태스크 | MNIST 이미지 10-class 분류 |
| 실험 목적 | SimpleConvNet과 DeepConvNet의 정확도 및 계산비용 차이를 재현한다. |
| 사전 가설 | DeepConvNet은 더 높은 계산비용을 사용하지만 더 높은 test accuracy를 달성한다. |
| 독립변인 | SimpleConvNet, DeepConvNet 전체 architecture recipe |
| 고정변수 | MNIST split, Adam lr=.001, batch 100, 20 epochs, 동일 평가 방식 |
| 종속변인·관찰값 | accuracy, loss, parameter count, FLOPs/MACs, epoch time, throughput, peak memory, filter visualization |

## 2. 분석 계획

정확도와 비용을 함께 비교하며 순수 depth ablation이 아닌 recipe 비교로 해석한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

Simple은 test 약 98.96%, Deep은 99% 이상 범위를 반복 실험 편차 내에서 재현하고, 동일 학습 예산에서 유의미한 정확도 차이를 확인한다.

## 4. 조회할 원자 실행

`CNN-SIMPLE`, `CNN-DEEP`

## 5. 해석 제한

구조가 크게 다르므로 두 조건은 별도 실행 그룹이다.
