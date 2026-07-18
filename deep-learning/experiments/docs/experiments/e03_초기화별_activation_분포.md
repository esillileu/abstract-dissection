# e03. 초기화별 activation 분포

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e03` |
| 데이터·태스크 | 표준정규 합성 입력을 사용하는 5-layer forward probe |
| 실험 목적 | activation과 initializer 조합에 따른 신호 소실, 폭주, 포화, dead activation을 재현한다. |
| 사전 가설 | 작은 고정 scale은 신호를 소실시키고 sigmoid/std=1은 포화한다. Xavier와 He는 대응 activation에서 분산을 더 안정적으로 보존한다. |
| 독립변인 | activation 3종 x initializer 4종 |
| 고정변수 | 입력 `(1000,100)`, width 100, depth 5, bias 0 |
| 종속변인·관찰값 | 층별 mean, std, percentile, zero ratio, saturation ratio, histogram, 마지막/첫 층 std 비율 |

## 2. 분석 계획

조합별 histogram과 signal-retention curve를 비교하고 interaction을 기술한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

std=.01의 깊이별 분산 감소, sigmoid/std=1의 포화 증가, Xavier·He 대응 조합의 상대적 분산 유지가 나타난다.

## 4. 조회할 원자 실행

`ACT-*` 12개

## 5. 해석 제한

학습 성능이 아니라 forward signal propagation 진단이다.
