# g09. parameter_matched_nn

<!-- Domain: deepscratch1 -->

## 구조 서명

`mnist-mlp-489-100-v1`

## 묶음 기준

동일한 ParameterMatchedNN 구조에서 입력 픽셀의 배열 조건만 변경한다.

## 공통 실행 설정

Flatten-FC489-ReLU-FC100-ReLU-FC10; Adam .001; batch 100; 2 epochs

## 원자 조건

| Atomic run ID         | Override             |
| --------------------- | -------------------- |
| `NN-MATCHED`          | 원본 MNIST           |
| `NN-MATCHED-PERMUTED` | 고정 픽셀 순열 MNIST |

## 사용 실험

e08

## 실행 정책

- 확률적 조건은 `research_v1`의 고정된 10개 seed를 사용한다.
- 원본·순열 조건의 동일 seed는 초기 난수 원본과 데이터 순서를 공유한다.
- `NN-MATCHED-PERMUTED`는 `CNN-SIMPLE-PERMUTED`와 동일한 permutation을 사용한다.
- permutation vector와 hash를 resolved config, runtime metadata, artifact에 기록한다.
- 기존 `FINISHED` run key가 있으면 재사용한다.
- 이 그룹의 결과는 사용 실험 수와 무관하게 한 번만 생성한다.
