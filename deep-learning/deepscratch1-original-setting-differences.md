# deepscratch1와 책 원본의 그래프·실행 차이

비교 대상은 `01_deep-learning-from-base/WegraLee-deep-learning-from-scratch`의 ch06~ch08
실행 코드다. 모델 구조와 학습률, batch size, float64, 복원추출 설정처럼 현재 일치하는
항목은 생략한다. 이 문서는 **그래프가 책의 그림과 다르게 보일 수 있는 이유**를 우선 기록한다.

## 그래프에 직접 영향을 주는 차이

| 실험     | 책 원본 그래프                                                                               | deepscratch1 그래프                                                                                        | 그래프상 영향                                                                                    |
| -------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| e02, e04 | 2,000 update 각각의 update 후 batch loss를 저장한 뒤 `smooth_curve`로 선을 그림              | 2,000 update 각각의 update 후 raw loss에 같은 `smooth_curve`를 seed별 적용한 뒤 평균·범위 error bar를 그림 | loss 해상도·smoothing·marker 간격은 원본과 맞춘다. seed 집계 error bar만 추가된다.               |
| e03      | 한 번 실행한 activation histogram                                                            | 고정 seed로 만든 histogram과 metric artifact                                                               | 조건·histogram bin 수는 같지만 난수 생성기와 난수열이 달라 bin count가 같지 않다.                |
| e05      | update 1, 11, 21, …, 191 뒤 train accuracy를 기록; BN on/off는 서로 다른 초기 가중치 | 같은 update 위치에서 full train/test accuracy를 기록; BN on/off는 동일 초기화·batch 열을 공유 | 기록 위치는 맞췄다. 현재 plot은 BatchNorm 효과를 더 직접적으로 비교한다. |
| e06      | update 1, 4, 7, …, 601 뒤 train/test accuracy를 기록                                          | 같은 update 위치에서 full train/test accuracy를 기록                                                       | 기록 위치와 관측점 수를 맞췄다. |
| e07      | update 1, 4, 7, …, 901 뒤 train/test accuracy를 기록                                          | 같은 update 위치에서 full train/test accuracy를 기록                                                       | 기록 위치와 관측점 수를 맞췄다. |
| e08      | 대응하는 원본 실험 없음                                                                              | ParameterMatchedNN·SimpleConvNet의 원본/고정 pixel-permuted full-test accuracy curve                       | 공간적 배치 활용을 보는 확장 실험이다. 책의 CNN curve 재현으로 해석하지 않는다.                 |
| e09      | SimpleConvNet은 앞 1,000개 train/test 표본 accuracy graph를 그림; DeepConvNet 학습 스크립트는 graph가 없음 | g08의 SimpleConvNet과 g10의 DeepConvNet을 같은 full-test accuracy graph에 표시                              | 학습 recipe는 원본과 같지만 관측 집합과 공통 비교 graph는 확장이다.                               |

## 의도적으로 유지하는 그래프 차이: 시행 재현성

다음은 책의 한 번 실행을 그대로 복사하지 않는 대신, 개별 시행 재현성과 비교의 공정성을
확보하기 위해 유지한다.

| 항목                                 | deepscratch1 방식                                                       | 그래프상 의미                                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 고정 master seed 및 분리 seed stream | 같은 seed로 model initialization, batch, dropout을 재생성               | 같은 run의 curve를 재실행해 확인할 수 있다. 책의 unseeded 단일 curve와 난수열은 같지 않다.                |
| seed 반복과 집계                     | e02/e04/e05/e06/e07/e08/e09은 10 seed curve를 평균·min/max error bar로 그림 | 한 번의 우연한 곡선 대신 경향과 변동 범위를 보여 준다. 원본 단일 curve와 직접 겹쳐 비교하면 안 된다.      |
| e05 BN paired 초기화                 | BN on/off가 같은 affine 초기 가중치와 같은 복원추출 batch 열을 사용     | 두 curve의 차이를 초기화·데이터 순서가 아닌 BatchNorm 효과로 해석할 수 있다. 이 보장은 테스트로 고정했다. |

## 그래프 외 실행·출력 차이

- 책은 각 장의 독립 NumPy 구현과 matplotlib/pickle 출력을 사용한다. deepscratch1은
  `mlprosection` trainer, MLflow metric, NPZ checkpoint를 사용한다.
- 같은 seed를 주어도 난수 소비 순서와 부동소수점 연산 순서가 달라 parameter 및 metric의
  bit-for-bit 일치는 보장하지 않는다.
- e05는 원본에 없는 test accuracy와 runtime metric을, e02/e04는 원본 plot에 없는 full
  train/test epoch 평가를 추가로 저장한다. 이는 원본 그래프를 대체하는 값이 아니라 추가
  분석 지표다.
- e09의 SimpleConvNet은 별도 재학습하지 않고 `g08/CNN-SIMPLE` 시행을 재사용한다. e09는
  다른 실험 ID가 아니라 실행 그룹 `g08`·`g10`의 MLflow 결과만 조회한다.
- e09의 `98.96%`·`99.38%`는 실행 하이퍼파라미터나 성공/실패 분기가 아니라 final full-test
  accuracy를 해석할 때 쓰는 규정 성능 기준이다.

## 해석 원칙

책 그림과의 비교는 curve의 절대 좌표 일치보다 학습 패턴·상대적 순서·안정성으로 한다.
현재의 seed 관리, 반복 집계, e05 paired 조건은 개별 시행 재현성과 통제된 비교를 위해
유지한다.
