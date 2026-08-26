# DeepScratch analysis pipeline

`analyze` has three storage boundaries. Final results are deliberately not a
cache.

1. **Raw cache** materializes immutable MLflow inputs by tracking store and run
   ID. Metric histories live under `mlflow_raw`; downloaded files live under
   `mlflow_artifact`.
2. **Prepared analysis cache** records the renderer-facing values produced from
   those inputs. It contains selected and normalized runs in
   `analysis_input.json`, plus filtered histories, metric values, CSV rows, and
   materialized files under `prepared/<study>/<variant>`.
3. **Result output** contains the PNG and Markdown intended for people. Every
   invocation renders these files again from prepared analysis inputs.

The normal execution path still queries MLflow run metadata to select the
current finished attempts. If the selected run IDs and metric declarations are
unchanged, it restores the analysis input and renderer-facing prepared values;
rendering does not reload metric history or raw artifacts.

Refresh modes are intentionally asymmetric:

- no option: reuse raw and prepared caches, then render results;
- `--refresh analysis`: reuse raw inputs and rebuild all prepared analysis
  values before rendering;
- `--refresh`: refresh raw inputs, rebuild prepared analysis values, then
  render.

Renderer code should access run data only through `StudyAnalysisInput`.
Adding a direct MLflow call or opening an external raw path inside a renderer
bypasses the prepared cache and breaks replay.
