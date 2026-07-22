# g08. simple_convnet

<!-- Domain: deepscratch1 -->

## 구조 서명

`mnist-simpleconvnet-v1`

## 묶음 기준

동일한 SimpleConvNet 구조에서 입력의 공간적 배치 조건만 변경한다. 원본 MNIST 조건은 규정 성능 분석에서도 재사용한다.

## 공통 실행 설정

Conv30 5x5-ReLU-Pool-FC100-ReLU-FC10; Adam .001; batch 100; 2 epochs

## 원자 조건

| Atomic run ID         | Override             |
| --------------------- | -------------------- |
| `CNN-SIMPLE`          | 원본 MNIST           |
| `CNN-SIMPLE-PERMUTED` | 고정 픽셀 순열 MNIST |

## 사용 실험

e08, e09

## 실행 정책

- 확률적 조건은 `research_v1`의 고정된 10개 seed를 사용한다.
- 원본·순열 조건의 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- 순열은 별도의 고정 seed로 한 번 생성하고 train·test 및 모든 모델에 동일하게 적용한다.
- permutation vector와 hash를 resolved config, runtime metadata, artifact에 기록한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
