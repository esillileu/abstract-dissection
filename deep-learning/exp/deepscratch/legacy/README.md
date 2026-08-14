# Historical MLflow compatibility

Historical runs remain available in the MLflow namespaces `ds1`,
`ds1_original`, `ds2`, and `ds2_original`. There is no local legacy result
store or fixed-seed fallback; all active and historical result loading goes
through MLflow.

## Archive import and recovery

Archives are imported append-only into the exact historical namespace:

```text
ds1          -> ds1
ds1_original -> ds1_original
ds2          -> ds2
ds2_original -> ds2_original
```

Use `exp import-legacy deepscratch <ds1|ds2> --variant <variant> --input
<archive.zip>`. An identical payload is reused, a different payload with the
same run key is retained as `imported-alternate`, and a collision with a
running run is deferred. Imports never copy a historical run into a
`deepscratch.ds1` or `deepscratch.ds2` writer namespace.

The preserved DS2 original regression coordinate is e05 / `BETTER-RNNLM` /
seed 4, run `8b19fdcd874c4c38b6a6480dc865101c`. It remains in
`ds2_original`; recovery validates its artifact and checkpoint inventory in
place instead of cloning it.

To recover derived output on another machine, import the archive and rerun
`exp analyze`. Storage audit only handles current `.staging/exp` records.
