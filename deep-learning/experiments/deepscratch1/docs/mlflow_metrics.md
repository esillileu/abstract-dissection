# deepscratch1 MLflow metric contract

This document is the canonical metric contract for `deepscratch1`.  It
describes values emitted by the executors, rather than a generic task schema.
Metric histories use the step type shown in the name; final metrics use MLflow
step `0`.

## All runs

Every run records `final/status/{success,nan_detected,inf_detected,diverged}`
and `final/system/{total_updates,completed_epochs,samples_seen}`.  Runtime and
memory metrics are optional until the corresponding profiler measurement is
available.

## e01: optimizer toy

`step/opt/{x,y,objective,distance_to_optimum,step_distance,cumulative_path_length,x_direction_changed,y_direction_changed}`
uses the optimizer update index.  Final values are
`final/opt/{objective,distance_to_optimum,path_length,x_direction_changes,y_direction_changes}`.

## e02 and e04--e09: MNIST MLP and CNN

- `update/train/loss` is the interval-mean training loss at global update.
- `update/train/raw_loss` is the optional post-update batch loss.
- `eval/test/loss` is an interval evaluation on the configured MNIST test
  split.  `eval/valid/loss` is retained only as a legacy alias.
- `epoch/train/{loss,accuracy}` and `epoch/test/{loss,accuracy}` use completed
  epoch indices.
- `final/train/{loss,accuracy}` and `final/test/{loss,accuracy}` are the final
  completed evaluation values.

`book_epoch/*` is a compatibility axis for book-style graph evaluations; it
must not be treated as the ordinary epoch metric axis.

## e03: activation probe

Each layer index records
`layer/<NN>/activation/{mean,std,min,max,p01,p25,median,p75,p99,zero_ratio,saturation_ratio,nonfinite_ratio}`.
Final summary metrics are
`final/activation/{std_retention_ratio,mean_absolute_shift,max_saturation_ratio,max_zero_ratio}`.

## Known gaps

BatchNorm statistics, regularization/generalization metrics, weight norms, and
CNN inference throughput/latency are not emitted yet.  The generic runtime and
memory contract is also incomplete when the profiler does not provide a source
measurement.
