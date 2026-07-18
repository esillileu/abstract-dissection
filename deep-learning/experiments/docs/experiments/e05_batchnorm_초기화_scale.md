# e05. BatchNorm-초기화 scale

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e05` |
| 데이터·태스크 | MNIST train 앞 1,000개를 이용한 분류와 학습 안정성 진단 |
| 실험 목적 | BatchNorm이 초기 weight scale 민감도를 완화하고 정상 학습 가능한 범위를 넓히는지 확인한다. |
| 사전 가설 | BN on은 no-BN보다 넓은 scale 구간에서 안정적으로 학습하지만, 초기화가 이미 적절한 조건에서는 반드시 빠르지 않을 수 있다. |
| 독립변인 | BatchNorm on/off x logspace scale 16종 |
| 고정변수 | `784-[100x5]-10`, ReLU, SGD lr=.01, batch 100, 20 epochs |
| 종속변인·관찰값 | epoch accuracy curve, normalized AUC, 목표 accuracy 도달 epoch, final accuracy, 성공 scale 개수와 log 범위 |

## 2. 분석 계획

scale별 BN on/off paired difference와 BN x scale interaction을 분석한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

BN on의 성공 scale 개수와 log 범위가 no-BN보다 크며 curve 분포 차이가 반복 실험에서 확인된다.

## 4. 조회할 원자 실행

`BN-OFF-01..16`, `BN-ON-01..16`

## 5. 해석 제한

BN on/off는 레이어 구조가 달라 별도 실행 그룹으로 관리한다.
