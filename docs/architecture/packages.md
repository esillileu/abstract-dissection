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
* **Dependencies:** `typer`, `pyyaml`, `numpy`, `tqdm`, `matplotlib`, `psutil`.
* **Responsibilities:**
  * **Unified CLI Engine:** Dynamic discovery and registration of study plugins (`repro_core.cli`).
  * **Central Path Resolution:** `RuntimePaths` workspace management (`.staging`, `.cache`, `artifacts`, `data`, `references`).
  * **Pure Checkpoint Lifecycle:** `CheckpointManager` handles retention policy (`CheckpointRetentionPolicy`), generational naming, atomic staging, and pointer JSONs (`latest.json`, `best.json`, `final.json`) via minimal callable delegation (`save_fn`, `epoch_fn`, `step_fn`).
  * **Execution Protocol:** Spec parser interfaces, experiment context (`ExperimentContext`), progress reporting, execution runners (`Runner`, `run_config`). Module-scoped executor dispatch via `ExecutionDefinition.executor_module`.
  * **Analysis Foundations:** Cross-run metrics aggregation, confidence interval estimators, publication-grade plotting styles.
* **Non-Goals:** Does NOT know anything about deep learning models, autograd parameters, layers, neural network architectures, or MLflow tracking packages. Has **zero dependencies on `repro-mlflow`**.

---

### 3) `repro-mlflow` (`packages/repro-mlflow`)
* **Role:** MLflow integration, durable tracking, artifact upload/download pipelines, and lineage validation.
* **Dependencies:** `repro-core`, `mlflow>=2.10.0`.
* **Responsibilities:**
  * **Schema Enforcement:** Standardized tag and metric namespaces (`SchemaV1Run`).
  * **Durable Uploads:** Synchronous/asynchronous artifact and metric batching (`client.log_batch`, `client.log_artifact`).
  * **Verification:** Integrity checksum verification (`_verify_uploaded_manifest`) before marking runs as durable.
  * **Artifact Caching:** Transparent local caching of remote checkpoints and metric payloads (`.cache/mlflow_artifact/`).
  * **YAML Runner:** CLI run orchestration with metadata validation (`run_yaml`).
* **Non-Goals:** Does not operate MLflow infrastructure, transfer historical runs,
  or maintain server-side checkpoints.

---

## 2. Dependency Matrix & Zero-Dependency Invariant

| Source Package | `numpy` / `psutil` | `repro-core` | `repro-mlflow` | `deepscratch` | `studies/*` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`deepscratch`** | ✅ Allowed | ❌ **PROHIBITED** | ❌ **PROHIBITED** | — | ❌ **PROHIBITED** |
| **`repro-core`** | ✅ Allowed | — | ❌ **PROHIBITED** | ❌ **PROHIBITED** | ❌ **PROHIBITED** |
| **`repro-mlflow`** | ✅ Allowed | ✅ Allowed | — | ❌ **PROHIBITED** | ❌ **PROHIBITED** |
| **`studies/dlfs`** | ✅ Allowed | ✅ Allowed | ✅ Allowed | ✅ Allowed | — |

---

## 3. Automated Boundary Enforcement

Architecture boundaries are strictly tested across all packages in [`tests/test_deepscratch_architecture.py`](file:///home/esillileu/abstract-dissection/tests/test_deepscratch_architecture.py):

1. **`deepscratch` Zero-Dependency Check:**
   Scans all Python files in `packages/deepscratch/src` and asserts **0 occurrences** of `repro_core`, `repro_mlflow`, or `dlfs`.
2. **`repro-core` Engine Independence Check:**
   Scans all Python files in `packages/repro-core/src` and asserts **0 occurrences** of `deepscratch` or `dlfs`.
3. **`repro-core` Tracking Package Independence Check:**
   Scans all Python files in `packages/repro-core/src` and asserts **0 occurrences** of `repro_mlflow`.
4. **`repro-core/context/checkpoint.py` Decoupling Check:**
   Scans `checkpoint.py` and asserts **0 occurrences** of deep learning parameter or buffer manipulation tokens (`named_parameters`, `named_buffers`, `save_params_npz`, `load_params_npz`).
5. **`repro-mlflow` Independence Check:**
   Scans all Python files in `packages/repro-mlflow/src` and asserts **0 occurrences** of `deepscratch` or `dlfs`.
