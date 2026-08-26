# Storage Policy, Caching & Path Resolution

This document specifies the exact lifecycle, storage tiers, and path resolution policies enforced across the monorepo.

---

## 1. Storage Tiers & Data Lifecycle

```mermaid
flowchart TD
    subgraph Execution Phase [1. Live Execution]
        RUNNER[repro run / executor] -->|Write raw tensors & buffers| STAGING[".staging/ (Volatile Buffer)"]
        STAGING -->|log_artifact / log_batch| MLFLOW[(MLflow Tracking Server\nSingle Source of Truth)]
        STAGING -.->|Discard on finish| DISCARD((Purged))
    end

    subgraph Analysis Phase [2. Analysis & Reporting]
        MLFLOW -->|Download runs & payloads| CACHE[".cache/mlflow_artifact/\n(Derived Cache)"]
        CACHE -->|Read & compute curves| ANALYSIS_ENGINE[Analysis Orchestrator]
        ANALYSIS_ENGINE -->|Render paper figures & markdown| ARTIFACTS["artifacts/analysis/\n(Human-Readable Reports)"]
    end

    subgraph Datasets [3. Canonical Data]
        DATA["data/ (MNIST, PTB, Sequence)\nCanonical Datasets"] --> RUNNER
    end
```

### 1) `.staging/` (Volatile Staging — Fully Ephemeral)
* **Purpose:** High-throughput in-memory/scratch buffer during live training.
* **Contents:** Working tensors, temporary event queues, intermediate checkpoint buffers.
* **Policy:**
  * Once a run finishes and MLflow upload completes, `.staging/` entries are safe to delete.
  * Can be cleaned anytime with `rm -rf .staging` without data loss.

### 2) `.cache/` (Derived Cache — 100% Reconstructible)
* **Purpose:** Acceleration of repeated I/O and expensive normalization computations.
* **Contents:**
  * `.cache/mlflow_artifact/`: Local copy of downloaded MLflow checkpoints and binary metrics.
  * `.cache/exp/<study>/.../analysis/cache.json`: Signature-cached intermediate statistical analysis results.
* **Policy:**
  * Never committed to git.
  * Can be cleaned anytime with `rm -rf .cache`; re-running analysis automatically reconstructs it from MLflow.

### 3) `artifacts/` (Human-Facing Reports & Results)
* **Purpose:** Permanent, human-inspectable deliverables and publication materials.
* **Contents:**
  * `artifacts/analysis/<study>/<suite>/`:
    * Publication-grade comparison charts (`.png`, `.pdf`)
    * Final statistical summaries and tables (`summary.md`, `summary.csv`)
    * Detailed observations (`observations.csv`)
  * `artifacts/runs/<study>/<suite>/<run_id>/`:
    * Local-only execution fallback (used strictly when MLflow is offline/disabled with `tracking.enabled: false`).

### 4) `data/` (Canonical Datasets)
* **Purpose:** Common dataset storage shared across all studies and packages.
* **Contents:** MNIST (`.gz`, `.pkl`), PTB (`.txt`, `.npy`, `.pkl`), Sequence datasets (`.txt`).
* **Policy:** Ignored by Git. Cached locally after first download/build.

---

## 2. MLflow Server as Single Source of Truth

**All production run artifacts, checkpoints, manifests, and time-series metrics are uploaded to MLflow Server (`infra/mlflow`).**

* The project root filesystem does **NOT** store raw run dumps during production runs.
* MLflow stores:
  * Full run configurations and parameter hashes
  * Time-series metrics (loss, accuracy, perplexity, elapsed times)
  * Generational checkpoints:
    * `checkpoints/generations/<role>-epoch-<E>-update-<U>-<digest>/`:
      * `model_parameters.npz`
      * `model_buffers.npz`
      * `objective_parameters.npz`
      * `objective_buffers.npz`
      * `optimizer_state.pkl`
      * `trainer_state.pkl`
      * `rng_state.pkl`
      * `manifest.json`
    * Pointers: `latest.json`, `best.json`, `final.json`
  * Lineage manifests (`result_manifest.json`, `checkpoint_manifest.json`)
* Upload verification (`_verify_uploaded_manifest`) ensures that runs are only marked `result.durable_complete = true` when all artifacts are durable in MLflow.

---

## 3. 3-Tier Dataset Path Resolution Precedence

To allow `deepscratch` to run both as a standalone library and inside monorepo studies, dataset paths are resolved using a 3-tier fallback precedence:

```
Tier 1: Explicit Parameter (data_dir=...)
   │ (if None)
   ▼
Tier 2: Environment Variable ($DEEPSCRATCH_DATA_DIR)
   │ (if unset)
   ▼
Tier 3: Default Local Fallback (./data/<dataset>)
```

### Study-Level Path Injection (Dependency Injection)
Inside `studies/dlfs`, explicit dataset paths are injected from `repro-core`'s central resolver via translation adapters:

```python
# In studies/dlfs/src/dlfs/ds1/implemented/adapters/data.py
def load_ds1_mnist(*, flatten: bool = True, gpu: bool = False, paths=None):
    runtime_paths = paths or RuntimePaths.from_environment()
    data_dir = runtime_paths.dataset("mnist")
    return load_mnist(flatten=flatten, gpu=gpu, data_dir=data_dir)
```

---

## 4. Environment Variable Overrides

All storage roots can be overridden via environment variables for CI, remote cluster mounting, or high-performance scratch drives:

| Environment Variable | Default Path | Purpose |
| :--- | :--- | :--- |
| `REPRO_DATA_ROOT` | `./data` | Canonical dataset root |
| `REPRO_ARTIFACTS_ROOT` | `./artifacts` | Human-facing analysis deliverables |
| `REPRO_CACHE_ROOT` | `./.cache` | Reconstructible cache storage |
| `REPRO_STAGING_ROOT` | `./.staging` | Ephemeral scratch directory |
| `REPRO_REFERENCES_ROOT` | `./references` | Upstream vendored baselines |
