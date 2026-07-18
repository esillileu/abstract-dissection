# e06. Weight decay

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e06` |
| 데이터·태스크 | MNIST train 앞 300개와 공식 test를 사용하는 저데이터 분류 |
| 실험 목적 | L2 정규화 강도에 따른 과적합 억제와 underfitting 경계를 확인한다. |
| 사전 가설 | lambda=0은 높은 train accuracy와 큰 gap을 만들고, 중간 lambda는 gap을 줄이며 과도한 lambda는 underfitting을 유발한다. |
| 독립변인 | lambda=0, 1e-4, 1e-3, 1e-2, 1e-1 |
| 고정변수 | `784-[100x6]-10`, ReLU, He, SGD lr=.01, batch 100, 301 epochs, dropout off |
| 종속변인·관찰값 | train/test accuracy curve, generalization gap, weight norm, loss |

## 2. 분석 계획

lambda에 따른 curve와 gap의 dose-response를 보고한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

양수 lambda 중 적어도 하나가 baseline보다 median gap을 20% 이상 줄이고 test accuracy를 2%p 초과해 악화시키지 않는다.

## 4. 조회할 원자 실행

`REG-BASE`, `REG-WD-1E4`, `REG-WD-1E3`, `REG-WD-1E2`, `REG-WD-1E1`

## 5. 해석 제한

test 성능으로 최적 lambda 하나만 사후 선택하지 않고 sweep 전체를 보고한다.
