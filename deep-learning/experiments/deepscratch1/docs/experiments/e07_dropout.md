# e07. Dropout

<!-- Domain: deepscratch1 -->

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e07` |
| 데이터·태스크 | MNIST train 앞 300개와 공식 test를 사용하는 저데이터 분류 |
| 실험 목적 | dropout ratio에 따른 과적합 억제와 underfitting 경계를 확인한다. |
| 사전 가설 | 중간 dropout은 gap을 줄이고, 높은 ratio는 학습 속도와 train 성능을 낮춘다. |
| 독립변인 | ratio=0, .1, .2, .3, .5 |
| 고정변수 | `784-[100x6]-10`, ReLU, He, SGD lr=.01, batch 100, 301 epochs, weight decay 0 |
| 종속변인·관찰값 | train/test accuracy curve, generalization gap, 수렴 속도, loss |

## 2. 분석 계획

ratio에 따른 dose-response와 중간 ratio 대비 .5의 underfitting 여부를 분석한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

.1-.3 중 하나가 baseline gap을 20% 이상 줄이며 test accuracy를 유지 또는 개선한다.

## 4. 조회할 원자 실행

`REG-BASE`, `REG-DO-01`, `REG-DO-02`, `REG-DO-03`, `REG-DO-05`

## 5. 해석 제한

`REG-BASE`는 e06과 공유하지만 dropout 조건은 레이어가 추가되므로 별도 실행 그룹이다.
