# e13. Attention Seq2seq

## 1. 실험 정의

| 항목 | 내용 |
|---|---|
| 실험 ID | `e13` |
| 데이터·태스크 | `date.txt` character-level 날짜 형식 변환 |
| 실험 목적 | Attention이 fixed-length representation 병목을 완화하고 alignment를 학습하는지 확인한다. |
| 사전 가설 | Attention은 Vanilla와 Peeky보다 빠르게 높은 exact-match에 도달하고 해석 가능한 alignment를 만든다. |
| 독립변인 | Vanilla, Peeky, Attention |
| 고정변수 | reverse=true, embedding 16, hidden 256, batch 128, 10 epochs, Adam .001, clip 5 |
| 종속변인·관찰값 | exact-match, token accuracy, AUC, 목표 accuracy 도달 epoch, 시간, parameter, memory, attention map와 entropy |

## 2. 분석 계획

세 구조의 paired curve와 대표 attention alignment를 함께 분석한다.

### 필수 보고물

- seed별 원자료
- 평균, 표준편차, median, 95% CI
- normalized AUC와 목표 지표 도달 시점
- paired difference
- 실패율
- 대표 curve와 최종 요약표

## 3. 재현·달성 기준

Attention의 final median exact-match가 90% 이상이고 accuracy AUC가 세 조건 중 가장 높으며 관련 위치의 alignment가 관찰된다.

## 4. 조회할 원자 실행

`SEQD-VAN-REV`, `SEQD-PEEKY-REV`, `SEQD-ATTN-REV`

## 5. 해석 제한

attention map의 정성적 예시만으로 성능 주장을 대신하지 않는다.
