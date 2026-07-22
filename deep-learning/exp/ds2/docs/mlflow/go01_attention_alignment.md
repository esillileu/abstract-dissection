# GO01 attention-alignment MLflow 스키마

`observations/attention.csv`:

```text
example_id,decode_step,encoder_position,weight
```

`observations/attention_render.json`에는 source/target label, input reversal, 축 방향, y-axis inversion, color range를 기록한다. 예제의 source/target/prediction은 `observations/predictions.csv`에 둔다.

```text
example_id,source,target,prediction
```

params/tags: `group_id=GO01`, `atomic_run_id=ATTENTION-ALIGNMENT`, `source_checkpoint_sha256`, `example_selection_seed=1984`, `decode_policy=greedy`, `render_spec_sha256`.

attention weight 전체는 MLflow scalar로 전송하지 않고 CSV artifact로 보존한다.
