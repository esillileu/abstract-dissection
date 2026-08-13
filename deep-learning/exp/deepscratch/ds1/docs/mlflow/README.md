# MLflow 기록 스키마

구현자는 실행 종류에 맞는 스키마만 사용한다.

| 대상 | 스키마 |
| --- | --- |
| `GT01`–`GT08` 학습 시행 | [GT 공용 스키마](gt_common.md) |
| `GO01` optimizer trajectory 관찰 | [GO01 스키마](go01_optimizer_trajectory.md) |
| `GO02` activation 관찰 | [GO02 스키마](go02_activation_observation.md) |

MLflow는 scalar history의 조회 인덱스이고, artifact CSV가 완전한 raw record다. 분석용 final/best/AUC/평균/CI metric은 이 스키마에 기록하지 않는다.
