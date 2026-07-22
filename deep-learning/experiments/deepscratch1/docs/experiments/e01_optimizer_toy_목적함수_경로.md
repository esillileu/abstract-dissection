# e01. Optimizer 목적함수 경로

<!-- Domain: deepscratch1 -->

## 원본

『밑바닥부터 시작하는 딥러닝』 1권 원본 저장소 [`WegraLee-deep-learning-from-scratch`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/)의 [`ch06/optimizer_compare_naive.py`](../../../../01_deep-learning-from-base/WegraLee-deep-learning-from-scratch/ch06/optimizer_compare_naive.py)를 재현한다. 목적함수, 초기점, 네 optimizer와 30-step 궤적은 원본 조건을 따른다. 이 실험의 step별 지표와 반복 실행용 보고 형식은 실험 도메인에서 추가했다.

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e01` |
| 데이터·태스크 | 실제 데이터셋 없이 `f(x,y)=x^2/20+y^2`를 최적화한다. |
| 실험 목적 | 비등방성 곡면에서 optimizer별 이동 경로와 축별 진동 특성을 재현한다. |
| 사전 가설 | Momentum, AdaGrad, Adam은 SGD와 서로 다른 축별 진동 및 접근 경로를 보인다. |
| 독립변인 | SGD, Momentum, AdaGrad, Adam |
| 고정변수 | 초기점 `(-7,2)`, 해석적 gradient, 30 updates, float64 |
| 종속변인·관찰값 | step별 `(x,y)`, 목적함수, 최적점 거리, 누적 경로 길이, 축별 부호 변경, 방향 전환량 |

## 2. 분석 계획

공개 코드의 step별 좌표와 비교하고, 등고선 위 궤적과 진동 지표를 함께 제시한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

공개 코드 최종값과 절대오차 `<=1e-6`; step별 좌표도 같은 허용오차를 만족한다.

## 4. 조회할 원자 실행

`TOY-SGD`, `TOY-MOM`, `TOY-ADAGRAD`, `TOY-ADAM`

## 5. 해석 제한

결정적 canonical 재현이므로 seed 반복을 수행하지 않는다.
