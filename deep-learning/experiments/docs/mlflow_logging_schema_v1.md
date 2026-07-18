# MLflow Logging Schema v1

> 대상: 딥러닝 밑바닥 재현 프로젝트  
> 목적: 동일 원자 조건을 다시 실행하지 않고도 실행 조건, 코드, 데이터, 난수, 모델, 학습 결과, runtime, memory, profiling 결과를 완전히 복원한다.  
> 구현 대상: Codex가 이 문서만 보고 로거와 검증기를 구현한다.  
> 스키마 버전: `1`

---

# 1. 최종 실행 모델

```text
condition_parent
└── seed_trial x N

analysis
└── condition_parent와 seed_trial 조회
```

| Run type | 의미 | 실제 학습 | 상세 metric | checkpoint |
|---|---|---:|---:|---:|
| `condition_parent` | seed 제외 원자 조건 | X | child 집계만 | X |
| `seed_trial` | 특정 seed의 실제 시행 | O | O | O |
| `analysis` | 여러 조건의 통계·비교 | X | 분석 결과 | X |

불변 원칙:

1. Parent 하나는 seed를 제외한 완전한 원자 조건 하나다.
2. Child 하나는 parent 조건과 master seed 하나의 조합이다.
3. 동일 `run_key`의 `FINISHED` child가 있으면 재실행하지 않는다.
4. 실제 설정의 진실의 원천은 `config/resolved.json`이다.
5. 실제 곡선 원자료의 진실의 원천은 artifact CSV다.
6. 실패 run은 삭제하지 않는다.
7. profiling이 꺼져 있어도 runtime과 memory는 기록한다.
8. post-run validation을 통과하기 전에는 성공으로 종료하지 않는다.

---

# 2. 식별자와 중복 판정

## 2.1 식별자

| 이름 | 의미 | 예시 |
|---|---|---|
| `experiment_id` | 연구 질문·분석 단위 | `e02` |
| `execution_group_id` | 동일 구조와 실행 경로 묶음 | `g02` |
| `atomic_run_id` | 사람이 읽는 원자 조건 ID | `MLP-SGD-HE` |
| `recipe_id` | 공통 실행 recipe | `RC-MLP` |
| `structure_signature` | 모델·데이터 흐름 구조 서명 | `mnist-mlp-784-100x4-10-relu-v1` |

다음 변경은 `structure_signature`를 변경한다.

- 레이어 추가·삭제·순서 변경
- hidden/embedding/channel dimension 변경
- recurrent cell 변경
- attention, peeky, weight tying 추가·삭제
- 출력 objective 그래프 변경
- 입력 또는 출력 표현 변경

다음 변경은 signature를 바꾸지 않는다.

- optimizer, learning rate
- initializer
- weight decay
- dropout ratio
- batch size, epoch 수
- seed

단, dropout 레이어 자체가 없는 구조와 있는 구조는 서로 다른 signature다.

## 2.2 Canonical JSON

해시 입력 JSON은 다음 규칙을 따른다.

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
)
```

## 2.3 Condition key

```text
condition_key = SHA256(canonical_condition_config)
```

포함:

- schema version
- atomic/execution group/recipe/structure identifiers
- code commit
- dataset config와 digest
- model, initializer, optimizer, scheduler
- train, eval, numerics, checkpoint
- profiling config

제외:

- 모든 seed
- MLflow run ID
- 시작·종료 시각
- hostname, process ID
- CUDA device index
- 측정된 runtime·memory 값

## 2.4 Run key

```text
run_key = SHA256(canonical_condition_config + canonical_seed_config)
```

## 2.5 Analysis key

```text
analysis_key = SHA256(
    experiment_id
    + 정렬된 source run key 목록
    + analysis config
    + analysis code commit
)
```

---

# 3. Run 이름

```text
Parent   : <atomic_run_id>
Child    : <atomic_run_id>-s<seed:02d>
Analysis : <experiment_id>-analysis-v<version>
```

예:

```text
MLP-SGD-HE
MLP-SGD-HE-s03
e02-analysis-v1
```

---

# 4. Tag 스키마

Tag는 검색, 필터링, 관계 표현에 사용한다.

## 4.1 모든 run 필수 tag

| Key | Type | 설명 |
|---|---|---|
| `schema.version` | string | `1` |
| `project.name` | string | 프로젝트 이름 |
| `run.type` | enum | `condition_parent`, `seed_trial`, `analysis` |
| `code.git_commit` | string | 40자리 commit SHA |
| `code.git_branch` | string | branch |
| `code.git_dirty` | bool-string | `true`, `false` |
| `code.repository` | string | repository 이름 |
| `code.entrypoint` | string | 실행 module 또는 script |
| `code.runner_version` | string | runner 버전 |
| `runtime.backend` | enum | `numpy`, `cupy` |
| `runtime.device_type` | enum | `cpu`, `cuda` |
| `runtime.platform` | string | 예: `linux` |
| `runtime.python_version` | string | 예: `3.11.14` |

## 4.2 Parent 필수 tag

```text
atomic_run.id
execution_group.id
recipe.id
structure.signature
condition.key
condition.status
condition.expected_children
dataset.id
model.family
task.type
consumer_experiment_ids
```

`condition.status`:

```text
planned | running | complete | partial | failed
```

## 4.3 Child 필수 tag

```text
atomic_run.id
execution_group.id
recipe.id
structure.signature
condition.key
run.key
master_seed
dataset.id
model.family
task.type
trial.status
trial.attempt
retry.of
parent.mlflow_run_id
```

`trial.status`:

```text
running | finished | failed | killed
```

## 4.4 Analysis 필수 tag

```text
experiment.id
analysis.key
analysis.version
analysis.status
analysis.paired
analysis.primary_metric
analysis.ci_method
analysis.multiple_comparison
source.manifest_digest
source.count
```

---

# 5. Param 스키마

Param은 실행 전에 결정되고 실행 중 변경되지 않는 값만 기록한다. 적용되지 않는 필드는 생략한다.

## 5.1 Policy

```text
policy/seed_count
policy/seed_start
policy/paired_execution
policy/save_final_checkpoint
policy/save_best_checkpoint
policy/fail_on_nan
policy/fail_on_inf
policy/fail_on_missing_metric
policy/resume_allowed
policy/retry_allowed
```

## 5.2 Seed

```text
seed/master
seed/model_init
seed/batch_order
seed/dropout
seed/negative_sampling
seed/synthetic_input
seed/dataset_split
seed/worker
```

고정 파생:

```text
model_init        = master_seed
batch_order       = master_seed + 10000
dropout           = master_seed + 20000
negative_sampling = master_seed + 30000
synthetic_input   = master_seed + 40000
worker            = master_seed + 50000
```

## 5.3 Dataset

```text
dataset/id
dataset/name
dataset/version
dataset/source
dataset/train_size
dataset/valid_size
dataset/test_size
dataset/input_shape
dataset/target_shape
dataset/num_classes
dataset/vocab_size

dataset/normalization
dataset/flatten
dataset/channel_order
dataset/tokenization
dataset/window_size
dataset/sequence_length
dataset/reverse_input

dataset/subset_policy
dataset/split_policy
dataset/split_seed
dataset/shuffle_before_split

dataset/source_digest
dataset/train_digest
dataset/valid_digest
dataset/test_digest
dataset/split_digest
dataset/subset_digest
dataset/vocab_digest
dataset/preprocessing_digest
```

## 5.4 Loader

```text
loader/batch_size
loader/shuffle
loader/drop_last
loader/sampling_method
loader/num_workers
loader/prefetch_factor
loader/pin_memory
loader/batch_order_seed
loader/steps_per_epoch
loader/samples_per_epoch
```

## 5.5 Model

```text
model/name
model/family
model/version
model/input_shape
model/output_shape
model/hidden_sizes
model/embedding_size
model/hidden_size
model/num_hidden_layers
model/num_recurrent_layers
model/num_conv_layers
model/activation
model/output_activation
model/normalization
model/use_batchnorm
model/use_dropout
model/dropout_ratio
model/use_weight_tying
model/use_attention
model/use_peeky
model/recurrent_cell
model/bidirectional
model/stateful
model/parameter_count
model/trainable_parameter_count
model/non_trainable_parameter_count
model/layer_count
model/structure_digest
model/structure_signature
model/dtype
model/compute_dtype
model/parameter_dtype
```

## 5.6 Initializer

```text
initializer/name
initializer/distribution
initializer/scale
initializer/fan_mode
initializer/gain
initializer/bias_value
initializer/embedding_std
initializer/recurrent_std
initializer/seed
```

## 5.7 Optimizer

```text
optimizer/name
optimizer/learning_rate
optimizer/momentum
optimizer/nesterov
optimizer/beta1
optimizer/beta2
optimizer/eps
optimizer/rho
optimizer/weight_decay
optimizer/weight_decay_mode
optimizer/dampening
```

## 5.8 Scheduler

```text
scheduler/name
scheduler/monitor
scheduler/mode
scheduler/factor
scheduler/patience
scheduler/min_lr
scheduler/cooldown
scheduler/step_size
scheduler/gamma
scheduler/warmup_steps
```

스케줄러가 없으면 `scheduler/name=constant`.

## 5.9 Loss와 regularization

```text
loss/name
loss/reduction
loss/label_smoothing
regularization/l2_lambda
regularization/l2_apply_to
regularization/l2_loss_factor
regularization/dropout_ratio
regularization/dropout_mode
regularization/dropout_locations
```

## 5.10 Training

```text
train/max_epochs
train/max_updates
train/batch_size
train/bptt_length
train/teacher_forcing
train/gradient_accumulation_steps
train/max_grad
train/gradient_clip_type
train/update_rule
train/eval_interval_unit
train/eval_interval
train/log_interval_unit
train/log_interval
train/early_stopping
train/early_stopping_metric
train/early_stopping_patience
train/state_reset_policy
train/sequence_decode_method
```

## 5.11 Evaluation

```text
eval/batch_size
eval/use_full_train
eval/use_full_valid
eval/use_full_test
eval/checkpoint_selection
eval/primary_metric
eval/smoothing_method
eval/smoothing_window
eval/decode_method
eval/max_decode_length
eval/exclude_start_token
eval/exact_match_policy
eval/threshold_name
eval/threshold_value
```

## 5.12 Numerics

```text
numerics/dtype
numerics/compute_dtype
numerics/parameter_dtype
numerics/accumulator_dtype
numerics/backend
numerics/device
numerics/deterministic
numerics/cudnn_deterministic
numerics/cudnn_benchmark
numerics/allow_tf32
numerics/nan_policy
numerics/inf_policy
numerics/epsilon
```

## 5.13 Checkpoint

```text
checkpoint/save_final
checkpoint/save_best
checkpoint/monitor
checkpoint/mode
checkpoint/interval_unit
checkpoint/interval
checkpoint/keep_last
checkpoint/format
checkpoint/include_optimizer
checkpoint/include_scheduler
checkpoint/include_rng_state
checkpoint/include_training_state
```

---

# 6. Profiling 설정과 기본 동작

## 6.1 Profiling param

```text
profiling/enabled
profiling/python_enabled
profiling/nsight_enabled
profiling/warmup_steps
profiling/profile_steps
profiling/record_shapes
profiling/record_memory
profiling/sample_interval_ms
profiling/export_python_prof
profiling/export_nsight_report
profiling/export_summary_json
```

권장 기본값:

```text
profiling/enabled=false
profiling/python_enabled=false
profiling/nsight_enabled=false
profiling/warmup_steps=5
profiling/profile_steps=20
profiling/record_shapes=false
profiling/record_memory=true
profiling/sample_interval_ms=100
profiling/export_python_prof=true
profiling/export_nsight_report=true
profiling/export_summary_json=true
```

## 6.2 `enabled=false`에서도 필수 기록

```text
runtime/train_total_s
runtime/epoch_time_s
runtime/throughput_samples_per_s

memory/cpu_rss_start_bytes
memory/cpu_rss_end_bytes
memory/cpu_rss_peak_sampled_bytes

memory/gpu_used_start_bytes
memory/gpu_used_end_bytes
memory/gpu_used_peak_sampled_bytes
memory/gpu_reserved_peak_sampled_bytes
```

GPU가 없으면 GPU metric은 생략한다.

## 6.3 `enabled=true`에서 추가 기록

```text
profile/forward/*
profile/backward/*
profile/update/*
profile/gradient_clip/*
```

각 phase suffix:

```text
total_s
mean_s
median_s
p95_s
std_s
min_s
max_s
count
fraction_of_train_time
```

예:

```text
profile/forward/total_s
profile/forward/p95_s
profile/forward/count
profile/forward/fraction_of_train_time
```

Gradient clipping이 비활성화된 경우:

```text
profile/gradient_clip/count=0
profile/gradient_clip/total_s=0
```

## 6.4 Runtime 정의

| Metric | 정의 |
|---|---|
| `runtime/train_total_s` | 첫 train update 직전부터 마지막 train update 직후까지 |
| `runtime/epoch_time_s` | 해당 epoch의 train 구간 |
| `runtime/throughput_samples_per_s` | 처리 sample 수 / train 구간 시간 |
| `runtime/eval_total_s` | 전체 평가 시간 |
| `runtime/checkpoint_total_s` | checkpoint 직렬화·저장 시간 |
| `runtime/artifact_logging_total_s` | MLflow artifact 기록 시간 |
| `runtime/run_wall_total_s` | run 시작부터 종료까지 wall time |

`train_total_s`에는 평가, checkpoint, artifact upload를 포함하지 않는다.

## 6.5 Memory 정의

CPU:

- `start`: 첫 train update 직전 RSS
- `end`: 최종 평가와 checkpoint 저장 직후 RSS
- `peak_sampled`: run 중 sampler가 관찰한 최대 RSS

GPU:

- `used`: 실제 사용 중인 GPU memory
- `reserved`: allocator가 예약한 memory
- `start`: 첫 train update 직전
- `end`: 최종 checkpoint 저장 직후
- `peak_sampled`: sampler 또는 backend peak API 기준 최대값

---

# 7. Metric 스키마

## 7.1 Step 의미

| Prefix | MLflow step |
|---|---|
| `update/*` | global update index |
| `epoch/*` | completed epoch index |
| `eval/*` | evaluation count |
| `runtime/epoch_time_s` | epoch index |
| `final/*`, `memory/*`, `profile/*` | 최종 step 0 |

동일 metric 이름에 update와 epoch 값을 섞지 않는다.

## 7.2 모든 학습 run 공통 update metric

```text
update/train/loss
update/train/data_loss
update/train/regularization_loss
update/optimizer/learning_rate
update/grad/global_norm_before_clip
update/grad/global_norm_after_clip
update/grad/clip_scale
update/grad/was_clipped
update/weight/global_norm
update/weight/update_norm
update/weight/update_ratio
update/runtime/step_s
update/runtime/throughput_samples_per_s
```

## 7.3 Epoch metric

```text
epoch/train/loss
epoch/train/accuracy
epoch/valid/loss
epoch/valid/accuracy
runtime/epoch_time_s
runtime/throughput_samples_per_s
```

## 7.4 Final 공통 metric

```text
final/train/loss
final/train/accuracy
final/valid/loss
final/valid/accuracy
final/test/loss
final/test/accuracy

runtime/train_total_s
runtime/eval_total_s
runtime/checkpoint_total_s
runtime/artifact_logging_total_s
runtime/run_wall_total_s

memory/cpu_rss_start_bytes
memory/cpu_rss_end_bytes
memory/cpu_rss_peak_sampled_bytes
memory/gpu_used_start_bytes
memory/gpu_used_end_bytes
memory/gpu_used_peak_sampled_bytes
memory/gpu_reserved_peak_sampled_bytes

final/status/success
final/status/nan_detected
final/status/inf_detected
final/status/diverged
final/system/total_updates
final/system/completed_epochs
final/system/samples_seen
```

bool metric은 `0.0` 또는 `1.0`으로 기록한다.

---

# 8. 태스크별 metric

## 8.1 Optimizer toy

```text
step/opt/x
step/opt/y
step/opt/objective
step/opt/distance_to_optimum
step/opt/step_distance
step/opt/cumulative_path_length
step/opt/turn_angle
step/opt/x_direction_changed
step/opt/y_direction_changed
final/opt/objective
final/opt/distance_to_optimum
final/opt/path_length
final/opt/x_direction_changes
final/opt/y_direction_changes
final/opt/mean_turn_angle
```

## 8.2 Activation probe

레이어별:

```text
layer/01/activation/mean
layer/01/activation/std
layer/01/activation/min
layer/01/activation/max
layer/01/activation/p01
layer/01/activation/p25
layer/01/activation/median
layer/01/activation/p75
layer/01/activation/p99
layer/01/activation/zero_ratio
layer/01/activation/saturation_ratio
layer/01/activation/nonfinite_ratio
```

최종:

```text
final/activation/std_retention_ratio
final/activation/mean_absolute_shift
final/activation/max_saturation_ratio
final/activation/max_zero_ratio
```

## 8.3 BatchNorm

```text
epoch/bn/running_mean_norm
epoch/bn/running_var_mean
epoch/bn/gamma_norm
epoch/bn/beta_norm
final/train/time_to_accuracy
final/train/reached_accuracy_threshold
final/bn/running_mean_norm
final/bn/running_var_mean
```

## 8.4 Regularization

```text
epoch/train/generalization_gap
epoch/weight/global_norm
epoch/regularization/loss
final/generalization_gap
final/weight/global_norm
final/best_valid_accuracy
final/best_valid_epoch
```

## 8.5 CNN

```text
model/flops
model/macs
epoch/runtime/train_images_per_s
final/runtime/inference_images_per_s
final/runtime/inference_latency_mean_ms
final/runtime/inference_latency_median_ms
final/runtime/inference_latency_p95_ms
final/runtime/inference_latency_std_ms
```

## 8.6 Word2Vec

```text
update/train/raw_loss
update/train/normalized_loss
epoch/train/normalized_loss
epoch/runtime/tokens_per_s
epoch/runtime/context_predictions_per_s
final/eval/nearest_neighbor_overlap
final/eval/analogy_top1_accuracy
final/eval/analogy_top5_accuracy
final/eval/embedding_norm_mean
final/eval/embedding_norm_std
```

## 8.7 Language model

```text
update/train/cross_entropy
update/train/ppl
epoch/train/ppl
epoch/valid/ppl
final/best_valid_ppl
final/best_valid_epoch
final/test/ppl
final/test/cross_entropy
```

## 8.8 Seq2seq

```text
epoch/train/token_accuracy
epoch/train/exact_match
epoch/valid/token_accuracy
epoch/valid/exact_match
final/test/token_accuracy
final/test/exact_match
final/best_valid_exact_match
final/best_valid_epoch
final/decode/samples_per_s
final/decode/mean_sequence_s
```

Attention:

```text
final/attention/entropy_mean
final/attention/entropy_std
final/attention/max_weight_mean
final/attention/diagonal_alignment_score
```

---

# 9. Curve summary

Loss:

```text
summary/train/loss/normalized_auc
summary/train/loss/first_window_mean
summary/train/loss/last_window_mean
summary/train/loss/relative_reduction
summary/train/loss/minimum
summary/train/loss/minimum_step
summary/train/loss/time_to_threshold
summary/train/loss/threshold_reached
```

Accuracy:

```text
summary/valid/accuracy/normalized_auc
summary/valid/accuracy/final
summary/valid/accuracy/maximum
summary/valid/accuracy/maximum_epoch
summary/valid/accuracy/time_to_threshold
summary/valid/accuracy/threshold_reached
```

PPL:

```text
summary/valid/ppl/normalized_auc
summary/valid/ppl/final
summary/valid/ppl/minimum
summary/valid/ppl/minimum_epoch
summary/valid/ppl/time_to_threshold
summary/valid/ppl/threshold_reached
```

Exact match:

```text
summary/valid/exact_match/normalized_auc
summary/valid/exact_match/final
summary/valid/exact_match/maximum
summary/valid/exact_match/maximum_epoch
summary/valid/exact_match/time_to_threshold
summary/valid/exact_match/threshold_reached
```

---

# 10. Artifact 구조

```text
artifacts/
├── config/
│   ├── resolved.json
│   ├── condition.json
│   ├── seed.json
│   ├── overrides.json
│   └── profiling.json
├── code/
│   ├── git.json
│   ├── git.diff.patch
│   └── entrypoint.py
├── environment/
│   ├── python.txt
│   ├── packages.txt
│   ├── system.json
│   ├── backend.json
│   └── device.json
├── data/
│   ├── dataset_manifest.json
│   ├── split_manifest.json
│   ├── preprocessing.json
│   ├── subset_indices.npy
│   └── vocabulary.json
├── model/
│   ├── architecture.json
│   ├── structure.txt
│   ├── parameter_manifest.json
│   └── initialization_manifest.json
├── metrics/
│   ├── history.csv
│   ├── update_history.csv
│   ├── epoch_history.csv
│   ├── runtime_history.csv
│   ├── memory_history.csv
│   └── final.json
├── checkpoints/
│   ├── final.*
│   ├── best.*
│   └── checkpoint_manifest.json
├── rng/
│   └── initial_rng_state.*
├── evaluation/
│   ├── predictions.*
│   ├── targets.*
│   ├── sample_results.csv
│   └── evaluation_summary.json
├── profiles/
│   ├── python.prof
│   ├── nsight_report.nsys-rep
│   └── profiling_summary.json
├── plots/
│   └── ...
└── failure/
    ├── exception.txt
    ├── traceback.txt
    └── failure_context.json
```

## 10.1 항상 필수

```text
config/resolved.json
config/condition.json
config/seed.json
config/profiling.json
code/git.json
environment/python.txt
environment/packages.txt
environment/system.json
data/dataset_manifest.json
model/architecture.json
model/parameter_manifest.json
metrics/history.csv
metrics/runtime_history.csv
metrics/memory_history.csv
metrics/final.json
checkpoints/checkpoint_manifest.json
```

## 10.2 조건부 필수

| 조건 | Artifact |
|---|---|
| dirty code | `code/git.diff.patch` |
| Python profiler | `profiles/python.prof` |
| Nsight | `profiles/nsight_report.nsys-rep` |
| profiling enabled | `profiles/profiling_summary.json` |
| 실패 | `failure/*` 3종 |
| attention | attention matrix와 plot |
| Word2Vec | embedding, vocab, analogy 결과 |
| best checkpoint | `checkpoints/best.*` |

---

# 11. 핵심 artifact 내용

## 11.1 `config/resolved.json`

```json
{
  "schema_version": 1,
  "atomic_run_id": "MLP-SGD-HE",
  "execution_group_id": "g02",
  "recipe_id": "RC-MLP",
  "structure_signature": "mnist-mlp-784-100x4-10-relu-v1",
  "dataset": {},
  "loader": {},
  "model": {},
  "initializer": {},
  "optimizer": {},
  "scheduler": {},
  "training": {},
  "evaluation": {},
  "numerics": {},
  "checkpoint": {},
  "profiling": {},
  "seed": {},
  "runtime": {}
}
```

## 11.2 `code/git.json`

```json
{
  "repository": "abstract-dissection",
  "commit": "<sha>",
  "branch": "main",
  "dirty": false,
  "remote": "<remote identifier>",
  "entrypoint": "experiments/run.py"
}
```

Dirty면 `git.diff.patch`가 필수다.

## 11.3 `environment/system.json`

최소 포함:

```text
OS 이름과 버전
kernel
CPU 모델
physical/logical core
RAM
GPU 모델과 VRAM
NVIDIA driver
CUDA runtime
NumPy/CuPy 버전
WSL 여부
container 여부
```

## 11.4 `model/parameter_manifest.json`

각 초기 parameter에 대해:

```text
name
shape
dtype
requires_grad
numel
initial_mean
initial_std
initial_min
initial_max
initial_norm
initial_digest
```

초기 digest로 paired initialization 적용 여부를 검증한다.

## 11.5 `metrics/history.csv`

Long format:

```csv
run_key,step_type,step,metric,value,timestamp
abc,update,0,train/loss,2.3025,...
abc,update,0,grad/global_norm_before_clip,1.352,...
abc,epoch,0,valid/accuracy,0.912,...
```

## 11.6 `metrics/runtime_history.csv`

```csv
step_type,step,train_s,eval_s,checkpoint_s,throughput_samples_per_s
epoch,0,12.3,1.4,0.2,4876.2
```

## 11.7 `metrics/memory_history.csv`

```csv
timestamp_s,cpu_rss_bytes,gpu_used_bytes,gpu_reserved_bytes
0.0,123456789,0,0
0.1,124000000,0,0
```

## 11.8 `profiles/profiling_summary.json`

```json
{
  "schema_version": 1,
  "enabled": true,
  "profiled_update_range": {"start": 100, "end": 119},
  "phases": {
    "forward": {
      "total_s": 1.2,
      "mean_s": 0.06,
      "median_s": 0.058,
      "p95_s": 0.071,
      "std_s": 0.006,
      "min_s": 0.052,
      "max_s": 0.076,
      "count": 20,
      "fraction_of_train_time": 0.41
    },
    "backward": {},
    "update": {},
    "gradient_clip": {}
  },
  "memory": {},
  "runtime": {},
  "tools": {
    "python_profiler": null,
    "nsight_systems": null
  }
}
```

## 11.9 `checkpoints/checkpoint_manifest.json`

```json
{
  "format": "pickle",
  "final": {
    "path": "final.pkl",
    "epoch": 20,
    "update": 9380,
    "digest": "<sha256>"
  },
  "best": {
    "path": "best.pkl",
    "metric": "valid/accuracy",
    "mode": "max",
    "value": 0.982,
    "epoch": 17,
    "digest": "<sha256>"
  },
  "contains": {
    "model": true,
    "optimizer": true,
    "scheduler": true,
    "rng_state": true,
    "training_state": true
  }
}
```

---

# 12. Parent 집계

상태:

```text
aggregate/children/expected
aggregate/children/finished
aggregate/children/failed
aggregate/children/killed
aggregate/children/missing
aggregate/children/success_rate
```

모든 주요 scalar metric에 대해:

```text
aggregate/<metric>/count
aggregate/<metric>/mean
aggregate/<metric>/std
aggregate/<metric>/median
aggregate/<metric>/min
aggregate/<metric>/max
aggregate/<metric>/ci95_low
aggregate/<metric>/ci95_high
```

Parent artifact:

```text
aggregate/
├── child_manifest.json
├── seed_results.csv
├── scalar_summary.csv
├── curve_summary.csv
├── runtime_summary.csv
├── memory_summary.csv
├── profiling_summary.csv
├── aggregate_summary.json
├── mean_curve.csv
└── plots/
```

---

# 13. Analysis run

Params:

```text
analysis/experiment_id
analysis/version
analysis/primary_metric
analysis/secondary_metrics
analysis/paired
analysis/seed_alignment
analysis/ci_method
analysis/confidence_level
analysis/bootstrap_samples
analysis/multiple_comparison
analysis/alpha
analysis/curve_interpolation
analysis/failure_policy
analysis/missing_pair_policy
```

권장 기본값:

```text
analysis/paired=true
analysis/seed_alignment=master_seed
analysis/ci_method=paired_bootstrap
analysis/confidence_level=0.95
analysis/bootstrap_samples=10000
analysis/multiple_comparison=holm
analysis/alpha=0.05
analysis/missing_pair_policy=exclude_pair_and_report
```

비교 metric:

```text
comparison/<A>_vs_<B>/paired_difference_mean
comparison/<A>_vs_<B>/paired_difference_median
comparison/<A>_vs_<B>/ci95_low
comparison/<A>_vs_<B>/ci95_high
comparison/<A>_vs_<B>/effect_size
comparison/<A>_vs_<B>/p_value
comparison/<A>_vs_<B>/adjusted_p_value
comparison/<A>_vs_<B>/win_rate
comparison/<A>_vs_<B>/criterion_met
```

Artifact:

```text
analysis/
├── config.json
├── source_manifest.json
├── seed_level_results.csv
├── aligned_curves.csv
├── condition_summary.csv
├── pairwise_differences.csv
├── statistical_tests.csv
├── acceptance_criteria.json
├── conclusion.json
└── plots/
```

---

# 14. 실패 처리

실패 tag:

```text
trial.status=failed
failure.type=nan | inf | oom | exception | metric_missing | artifact_missing
failure.stage=setup | data | forward | backward | update | evaluation | checkpoint | artifact
```

실패 metric:

```text
final/status/success=0
final/status/nan_detected
final/status/inf_detected
final/system/completed_epochs
final/system/total_updates
runtime/run_wall_total_s
memory/cpu_rss_peak_sampled_bytes
memory/gpu_used_peak_sampled_bytes
```

`failure/failure_context.json`:

```json
{
  "epoch": 3,
  "update": 1523,
  "last_finite_loss": 1.352,
  "last_gradient_norm": 923.5,
  "batch_indices": [],
  "device_memory": {},
  "exception_type": "",
  "exception_message": ""
}
```

---

# 15. Pre-run validation

다음 중 하나라도 실패하면 child를 시작하지 않는다.

```text
atomic_run.id 존재
execution_group.id 존재
recipe.id 존재
structure.signature 존재
schema.version 일치
resolved config 미정 필드 없음
알 수 없는 config key 없음
dataset digest 생성 완료
split/subset digest 생성 완료
model structure digest 생성 완료
parameter 수 계산 가능
seed 역할 전부 결정
condition key 계산 완료
run key 계산 완료
동일 FINISHED run key 없음
필수 metric schema 결정
checkpoint policy 결정
profiling config 결정
memory sampler 초기화 가능
```

Parent 생성 시:

```text
plan/
├── resolved_conditions.json
├── planned_children.json
└── validation_report.json
```

---

# 16. Post-run validation

`seed_trial`을 성공 종료하기 전에 검사한다.

```text
필수 tag 존재
필수 param 존재
필수 artifact 존재
metrics/final.json 존재
checkpoint manifest 존재
history.csv 파싱 가능
runtime/train_total_s 존재
runtime/epoch_time_s 최소 1개 존재
runtime/throughput_samples_per_s 존재
CPU RSS start/end/peak 존재
GPU 실행이면 GPU memory start/end/peak 존재
profiling enabled이면 profiling_summary.json 존재
Python profiler 사용이면 python.prof 존재
Nsight 사용이면 nsys-rep 존재
NaN/Inf 검사 통과
run key와 config digest 일치
checkpoint digest 검증 통과
```

실패 시:

```text
trial.status=failed
failure.type=metric_missing 또는 artifact_missing
```

---

# 17. 권장 Python 인터페이스

```python
@dataclass(frozen=True)
class RunIdentity:
    schema_version: int
    project_name: str
    experiment_ids: tuple[str, ...]
    atomic_run_id: str
    execution_group_id: str
    recipe_id: str
    structure_signature: str
    condition_key: str
    run_key: str
    master_seed: int


@dataclass(frozen=True)
class ProfilingConfig:
    enabled: bool = False
    python_enabled: bool = False
    nsight_enabled: bool = False
    warmup_steps: int = 5
    profile_steps: int = 20
    record_shapes: bool = False
    record_memory: bool = True
    sample_interval_ms: int = 100
    export_python_prof: bool = True
    export_nsight_report: bool = True
    export_summary_json: bool = True


class MLflowRunLogger:
    def start_parent(self, ...): ...
    def start_child(self, ...): ...
    def log_tags(self, tags: dict[str, str]): ...
    def log_params(self, params: dict[str, object]): ...
    def log_update_metrics(self, step: int, metrics: dict[str, float]): ...
    def log_epoch_metrics(self, epoch: int, metrics: dict[str, float]): ...
    def log_final_metrics(self, metrics: dict[str, float]): ...
    def log_artifact_tree(self, root: Path): ...
    def finalize_success(self): ...
    def finalize_failure(self, exc: BaseException): ...


class RuntimeMonitor:
    def start_run(self): ...
    def start_train(self): ...
    def start_epoch(self, epoch: int): ...
    def end_epoch(self, epoch: int, samples: int): ...
    def end_train(self, samples: int): ...
    def snapshot(self): ...
    def finalize(self) -> dict[str, float]: ...


class PhaseProfiler:
    def forward(self): ...
    def backward(self): ...
    def update(self): ...
    def gradient_clip(self): ...
    def summary(self) -> dict[str, dict[str, float]]: ...
```

---

# 18. 구현 순서

1. config dataclass와 JSON schema
2. canonical JSON과 digest 함수
3. condition/run/analysis key
4. flatten tag·param 함수
5. parent-child MLflow lifecycle
6. runtime timer
7. CPU RSS sampler
8. GPU memory sampler
9. profiling disabled 기본 metric
10. phase profiler
11. artifact tree writer
12. success/failure finalizer
13. pre-run validation
14. post-run validation
15. parent aggregate
16. analysis logger
17. 단위·통합 테스트

---

# 19. 테스트 요구사항

단위 테스트:

```text
canonical JSON key order 불변
같은 config는 같은 condition_key
seed만 다르면 condition_key 동일
seed가 다르면 run_key 다름
commit이 다르면 condition_key 다름
profiling config가 다르면 condition_key 다름
dirty인데 diff artifact 없으면 validation 실패
GPU가 없으면 GPU metric 생략
profiling false에서도 runtime/memory 존재
profiling true에서 phase summary 존재
failed run에서 failure artifact 존재
```

통합 테스트:

```text
parent 생성
child 2개 생성
동일 run_key 재실행 차단
runtime metric 기록
CPU memory 기록
GPU 사용 시 GPU memory 기록
profiling false 정상 종료
profiling true phase metric 기록
artifact tree 생성
checkpoint manifest 생성
parent aggregate 생성
```

---

# 20. 최종 불변 규칙

1. `resolved.json`이 전체 설정의 진실의 원천이다.
2. child 하나는 master seed 하나만 가진다.
3. 동일 run key는 한 번만 성공한다.
4. code commit은 condition key에 포함한다.
5. profiling config도 condition key에 포함한다.
6. profiling이 꺼져 있어도 runtime과 memory를 기록한다.
7. profiling이 켜져 있으면 phase metric과 profile artifact를 기록한다.
8. update, epoch, final metric의 step 의미를 섞지 않는다.
9. failed run을 삭제하지 않는다.
10. dirty 실행은 diff 없이는 허용하지 않는다.
11. dataset digest 없이는 실행하지 않는다.
12. 초기 parameter digest를 기록한다.
13. checkpoint에는 model, optimizer, scheduler, RNG, training state를 포함한다.
14. MLflow metric이 누락돼도 artifact 원자료는 남아야 한다.
15. post-run validation 통과 전에는 `FINISHED`로 종료하지 않는다.
