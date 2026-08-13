# GO02 activation observation MLflow 스키마

대상: `GO02`의 activation × initializer 12 조건.

## MLflow params/tags

```text
group_id=GO02
atomic_run_id
activation
initializer
input_shape=1000x100
width=100
depth=5
bias=0
input_seed
model_seed
resolved_config_sha256
```

## histogram artifact

한 조건은 forward pass를 한 번 수행한다. histogram image만 저장하지 않고 bin count를 저장한다.

`observations/activation_histogram.csv` 열:

```text
layer,bin_index,bin_left,bin_right,count,sample_count
```

## layer summary

`observations/activation_summary.csv` 열:

```text
layer,mean,std,min,max,zero_ratio,sample_count
```

각 layer `n=1..5`에 대해 다음 MLflow scalar를 기록한다.

```text
observation/activation/layer_{n}/mean
observation/activation/layer_{n}/std
observation/activation/layer_{n}/zero_ratio
```

MLflow step은 모든 layer summary에서 `0`이다. histogram 전체는 MLflow metric으로 전송하지 않고 CSV artifact만 전송한다.
