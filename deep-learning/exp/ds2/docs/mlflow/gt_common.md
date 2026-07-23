# GT 공용 MLflow 스키마

모든 `GT01`–`GT07` trial의 params/tags: `group_id`, `atomic_run_id`, `master_seed`, `model_signature`, `dataset_id`, `split_id`, `resolved_config_sha256`, `evaluation_schedule_id`, `source_curve_schedule_id`, `loss_phase=post_update`, `loss_reduction=mean`.

## MLflow metric 목록

- `final/status/*` - run 성공 여부와 NaN/Inf/divergence flag.
- `final/system/*` - 총 update 수, 완료 epoch 수, 처리 sample/token/sequence 수.
- `final/train/loss` - Word2Vec/Seq2seq final train loss.
- `final/train/book_loss` - Word2Vec의 마지막 post-update 책 objective.
- `final/train/perplexity`, `final/test/perplexity`, `final/{train,test}/ppl` - language modeling final perplexity.
- `final/valid/perplexity`, `final/valid/ppl`, `final/best_valid_ppl`, `final/best_valid_epoch` - validation-selected LM recipe summary. valid evaluation이 있을 때만 기록한다.
- `final/test/exact_match`, `final/test/token` - Seq2seq final exact-match/token accuracy projection.
- `runtime/run_wall_total_s` - YAML runner 기준 전체 run wall time.
- `runtime/train_total_s` - executor가 trainer fit 구간에서 측정한 전체 train wall time.
- `memory/cpu_rss_start_bytes`, `memory/cpu_rss_end_bytes`, `memory/cpu_rss_peak_sampled_bytes` - profiling summary에서 투영한 CPU RSS 시작/종료/peak sampled memory.
- `update/train/loss`, `update/train/lr` - 각 optimizer update 직후 표준 mean loss와 learning rate.
- `update/train/book_loss` - Word2Vec의 post-update sum-over-terms/mean-over-examples objective.
- `{epoch,terminal}/eval_{valid,test}/{loss,perplexity,exact_match_accuracy,token_accuracy}` - LM/Seq2seq evaluation schedule이 요청한 long-form evaluation result.
- `series/train/loss` - `observations/source_curves.csv`의 source objective loss. step은 `plot_index`.
- `series/train/book_loss` - Word2Vec 원본 graph용 pre-update interval book loss. step은 `plot_index`.
- `series/train/perplexity` - `observations/source_curves.csv`의 source objective perplexity. step은 `plot_index`.
- `series/eval_test/exact_match_accuracy` - Seq2seq epoch exact-match source curve. step은 `plot_index`.
- `update/runtime/window/{train,eval}_wall_time_ms` - timing window의 host wall time. step은 `end_update`.
- `update/runtime/window/{train,eval}_device_time_ms` - CUDA profile-mode device time. 값이 있을 때만 기록한다.

## MLflow artifact 목록

- `updates.csv` - update별 raw train loss, 선택적 book loss, lr history.
- `evaluations.csv` - split/metric long-form evaluation history.
- `timing_windows.csv` - source-curve/evaluation probe, epoch, terminal timing window history.
- `checkpoints.csv` - executor가 기록한 selected/eval checkpoint index와 digest.
- `observations/source_curves.csv` - 원본 graph/console series 재현용 source curve history.
- `observations/predictions.csv` - Seq2seq prediction examples.
- `observations/attention.csv`, `observations/attention_render.json` - attention alignment observation weights와 render metadata.
- `config/resolved.json`, `config/condition.json`, `config/seed.json`, `config/profiling.json` - resolved run condition, seed stream, profiling config.
- `reproducibility/runtime.json` - 실제 runtime/backend/seed/data metadata.
- `code/git.json`, `code/git.diff.patch` - git commit/dirty 상태와 dirty diff. diff는 dirty run에서만 기록한다.
- `environment/*.json`, `environment/*.txt` - Python, package, system, backend, device metadata.
- `data/dataset_manifest.json` - dataset section과 data selection metadata.
- `model/architecture.json`, `model/structure.txt`, `model/parameter_manifest.json`, `model/initialization_manifest.json` - model config, structure text, parameter manifest, initializer metadata.
- `metrics/metrics.csv`, `metrics/final.json` - MLflow metric row mirror와 final scalar metric snapshot.
- `metrics/runtime_history.csv`, `metrics/memory_history.csv` - profiling metrics에서 만든 runtime/memory history artifact.
- `profiles/profiling_summary.json` - `ExperimentResult.profiling_metrics` 전체 summary.
- `checkpoints/checkpoint_manifest.json` - checkpoint payload 위치, digest, 포함 state manifest.
- `checkpoints/latest.json + generations/latest-*/` - final checkpoint payload. `tracking.upload_checkpoint`가 true일 때 MLflow artifact로 업로드한다.

| artifact 열 | MLflow metric | step |
| --- | --- | ---: |
| `updates.csv.loss` | `update/train/loss` | update |
| `updates.csv.book_loss` | `update/train/book_loss` | update |
| `updates.csv.lr` | `update/train/lr` | update |
| `observations/source_curves.csv` loss | `series/train/loss` | `plot_index` |
| `observations/source_curves.csv` book loss | `series/train/book_loss` | `plot_index` |
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
