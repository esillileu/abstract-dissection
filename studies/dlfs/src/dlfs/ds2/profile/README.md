# DS2 direct profiling

`exp profile` is a diagnostic path that does not create MLflow runs. Formal,
comparable studies use `exp run`; the former e02 profiler was promoted to
`e10/PF01` and `e11/PF02` and is available only through that formal path.

Direct profiling remains available for custom or exploratory engines:

```bash
just exp profile deepscratch ds2 -e 05
just exp profile deepscratch ds2 -e 06
```

Default paths are resolved by `profile/paths.py` beneath `EXP_CACHE_ROOT` (or
`.cache`) and have distinct ownership:

- `.../<study>/implemented/profile/measurements`: profiler JSON and measurement payloads
- `.../<study>/implemented/profile/analysis`: derived tables, reports, and figures
- `.../<study>/implemented/profile/artifacts`: profiler-native files such as Nsight reports

An explicit `--output-dir` overrides only the measurement path for a one-off
diagnostic. It does not turn a direct profile into a canonical MLflow result.

The e05 Nsight helper writes reports to `artifacts/nsys` and its summaries to
`analysis/nsys`. Synchronization and repeated timing use the common profiling
APIs where the custom engine permits it.
