# Packages Architecture & Boundaries

The `packages/` directory houses independent, reusable Python libraries. Each package defines its own `pyproject.toml` and enforces strict boundary isolation.

---

## 1. Package Inventory & Responsibilities

### 1) `deepscratch` (`packages/deepscratch`)
* **Role:** Pure standalone mini deep-learning framework (Tensors, Autograd, Layers, Optimizers, Trainers, Datasets).
* **Dependencies:** `numpy>=1.26.0`, `psutil>=5.9.0` (Optional: `cupy` for GPU acceleration).
* **Strict Invariant (Zero-Dependency):**
  * **0 imports** of `repro_core`, `repro_mlflow`, or any study packages (`dlfs`).
  * Self-contained numerical backend abstraction (`deepscratch.core.backend`).
  * Self-contained event-driven training system (`deepscratch.trainer.events`).
  * Built-in 3-tier dataset loaders (`deepscratch.datasets`).
* **Submodules:**
  * `core/`: Tensor definitions, autograd nodes, device management, backend abstractions (`Backend`, `Array`, `make_backend`, `configure_runtime`).
  * `nn/`: Neural network layers (`linear`, `cnn`, `recurrent`, `time`), activations, objectives, sampling utilities.
  * `optim/`: Optimizers (`SGD`, `Momentum`, `Adam`, `AdaGrad`).
  * `trainer/`: Event-driven trainers (`EventTrainer`, `ForwardTrainer`, `LanguageModelTrainer`, `Seq2SeqTrainer`, `Word2VecTrainer`).
  * `datasets/`: Dataset loaders (`mnist`, `ptb`, `sequence`, `spiral`) with 3-tier path resolution.
  * `profiling/`: Device timers and timing contexts.

---

### 2) `repro-core` (`packages/repro-core`)
* **Role:** General-purpose scientific experiment orchestration and reproducibility infrastructure.
* **Dependencies:** `click`, `pyyaml`, `numpy`, `tabulate`, `matplotlib`.
* **Responsibilities:**
  * **Unified CLI Engine:** Dynamic discovery and registration of study plugins (`repro_core.cli`).
  * **Central Path Resolution:** `RuntimePaths` workspace management (`.staging`, `.cache`, `artifacts`, `data`, `references`).
  * **Execution Protocol:** Spec parser interfaces, experiment context (`ExperimentContext`), progress reporting, checkpoint retention policies (`CheckpointRetentionPolicy`).
  * **Analysis Foundations:** Cross-run metrics aggregation, confidence interval estimators, publication-grade plotting styles.
* **Non-Goals:** Does NOT know anything about deep learning models, layers, or neural network architectures.

---

### 3) `repro-mlflow` (`packages/repro-mlflow`)
* **Role:** MLflow integration, durable tracking, artifact upload/download pipelines, and lineage validation.
* **Dependencies:** `repro-core`, `mlflow>=2.10.0`.
* **Responsibilities:**
  * **Schema Enforcement:** Standardized tag and metric namespaces (`SchemaV1Run`).
  * **Durable Uploads:** Synchronous/asynchronous artifact and metric batching (`client.log_batch`, `client.log_artifact`).
  * **Verification:** Integrity checksum verification (`_verify_uploaded_manifest`) before marking runs as durable.
  * **Artifact Caching:** Transparent local caching of remote checkpoints and metric payloads (`.cache/mlflow_artifact/`).

---

## 2. Dependency Matrix & Zero-Dependency Invariant

| Source Package | `numpy` / `psutil` | `repro-core` | `repro-mlflow` | `deepscratch` | `studies/*` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`deepscratch`** | ✅ Allowed | ❌ **PROHIBITED** | ❌ **PROHIBITED** | — | ❌ **PROHIBITED** |
| **`repro-core`** | ✅ Allowed | — | ❌ Prohibited | ❌ Prohibited | ❌ Prohibited |
| **`repro-mlflow`** | ✅ Allowed | ✅ Allowed | — | ❌ Prohibited | ❌ Prohibited |
| **`studies/dlfs`** | ✅ Allowed | ✅ Allowed | ✅ Allowed | ✅ Allowed | — |

---

## 3. Automated Boundary Enforcement

Architecture boundaries are strictly tested in the test suite:

```python
# tests/test_deepscratch_architecture.py
def test_deepscratch_package_has_zero_dependencies_on_repro_core():
    # Recursively scans all .py files in packages/deepscratch/src
    # Asserts 0 occurrences of 'repro_core', 'repro_mlflow', or 'dlfs'
```
