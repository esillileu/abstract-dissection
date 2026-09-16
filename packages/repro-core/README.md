# repro-core

Tracking-neutral execution, paths, checkpoint delegation and result contracts.
`analysis.core` owns Curve, aggregate and pure plot helpers. It does not select
MLflow runs, interpret model parameter manifests or define book analysis policy.
Those policies belong to DLFS; MLflow clients and result loading belong to repro-mlflow.
See [package architecture](../../docs/architecture/packages.md).
