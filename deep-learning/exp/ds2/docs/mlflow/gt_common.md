# GT 공용 MLflow 스키마

모든 `GT01`–`GT07` trial의 params/tags: `group_id`, `atomic_run_id`, `master_seed`, `model_signature`, `dataset_id`, `split_id`, `resolved_config_sha256`, `evaluation_schedule_id`, `source_curve_schedule_id`, `loss_phase=post_update`, `loss_reduction=mean`.

| artifact 열 | MLflow metric | step |
| --- | --- | ---: |
| `updates.csv.loss` | `update/train/loss` | update |
| `updates.csv.lr` | `update/train/lr` | update |
| `observations/source_curves.csv` loss | `series/train/loss` | `plot_index` |
| `observations/source_curves.csv` PPL | `series/train/perplexity` | `plot_index` |
| `observations/source_curves.csv` exact-match | `series/eval_test/exact_match_accuracy` | `plot_index` |
| valid PPL | `epoch/eval_valid/perplexity` | epoch |
| test PPL | `terminal/eval_test/perplexity` | last update |
| sequence exact-match | `epoch/eval_test/exact_match_accuracy` | epoch |
| sequence token accuracy | `epoch/eval_test/token_accuracy` | epoch |

## Timing window 기록

executor는 source-curve/evaluation probe 간 학습 시간 또는 epoch/terminal에서 닫힌 학습 시간을 `timing_windows.csv`에 기록한다. Trainer가 직접 MLflow나 CSV를 쓰지 않는다.

| MLflow metric | step | artifact 열 |
| --- | ---: | --- |
| `runtime/window/train_wall_time_ms` | `end_update` | `train_wall_time_ns / 1_000_000` |
| `runtime/window/train_device_time_ms` | `end_update` | `train_device_time_ns / 1_000_000` |
| `runtime/window/eval_wall_time_ms` | `end_update` | `eval_wall_time_ns / 1_000_000` |
| `runtime/window/eval_device_time_ms` | `end_update` | `eval_device_time_ns / 1_000_000` |

device-time metric은 backend event profile mode에서만 보낸다. 일반 GPU 실행은 매 update 동기화 없이 window 종료에서만 시간을 확정한다.

MLflow는 scalar index이고 `updates.csv`, `evaluations.csv`, `timing_windows.csv`, `observations/source_curves.csv`가 완전 history다. `plot_index`는 원본 그래프의 x축이며 global update와 다르다. final/best/AUC/평균/CI는 기록하지 않는다.
