# GO01 optimizer trajectory MLflow 스키마

대상: `GO01`의 `TOY-SGD`, `TOY-MOMENTUM`, `TOY-ADAGRAD`, `TOY-ADAM`.

## MLflow params/tags

```text
group_id=GO01
atomic_run_id
objective_id=anisotropic_quadratic_v1
initial_x=-7.0
initial_y=2.0
max_updates=30
dtype=float64
resolved_config_sha256
```

## 관찰 record

각 update **전** state를 한 행 기록한다.

`observations/trajectory.csv` 열:

```text
update,x,y,objective,grad_x,grad_y
```

| MLflow metric | step | artifact 열 |
| --- | ---: | --- |
| `update/trajectory/x` | update | `x` |
| `update/trajectory/y` | update | `y` |
| `update/trajectory/objective` | update | `objective` |
| `update/trajectory/grad_x` | update | `grad_x` |
| `update/trajectory/grad_y` | update | `grad_y` |

30개 record 전체를 메모리에 쌓고 run 종료 시 CSV artifact와 MLflow batch로 한 번 전송한다.
