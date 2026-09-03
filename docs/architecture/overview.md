# Monorepo Architecture Overview

This repository (`abstract-dissection`) is a scientific reproduction monorepo designed for exact, verifiable, and transparent deep learning experiments.

---

## 1. 4-Tier Top-Level Architecture

The repository is organized into four distinct architectural layers with strict unidirectional dependency flow:

```mermaid
graph TD
    subgraph Layer 1: References [1. References (External Immutables)]
        REF[references/dlfs*-book, papers]
    end

    subgraph Layer 2: Core Packages [2. Packages (Independent Engines)]
        CORE[packages/repro-core]
        MLFLOW[packages/repro-mlflow]
        ENGINE[packages/deepscratch]
    end

    subgraph Layer 3: Studies [3. Studies (Scientific Experiments & Campaigns)]
        STUDY_DLFS[studies/dlfs\nDLFS Vol 1 & 2 Suite]
        STUDY_F2[studies/f2\nF2 Campaign: Corpus Pipeline + 8-10 Studies]
        ADAPTERS[studies/*/adapters\nTranslation Adapters Boundary]
    end

    subgraph Layer 4: Infrastructure [4. Infra & Tracking]
        INFRA[Externally managed MLflow and PostgreSQL services]
    end

    STUDY_DLFS -->|orchestrates via| ADAPTERS
    STUDY_F2 -->|orchestrates via| ADAPTERS
    ADAPTERS -->|uses pure retention| CORE
    ADAPTERS -->|injects datasets & builds models| ENGINE
    STUDY_DLFS -->|tracks runs| MLFLOW
    STUDY_F2 -->|tracks runs| MLFLOW
    STUDY_DLFS -->|reads immutable baseline| REF
    MLFLOW -->|depends on| CORE
    INFRA -.->|serves tracking & storage| MLFLOW
    INFRA -.->|metadata persistence| STUDY_F2
```

---

## 2. Directory Layout & Roles

| Top-Level Directory | Ownership / Role | Dependency Rules |
| :--- | :--- | :--- |
| **`packages/`** | Reusable, self-contained Python libraries (`repro-core`, `repro-mlflow`, `deepscratch`). | **Must NOT depend on `studies/` or external references.** `deepscratch` has 0 dependencies on other workspace packages. `repro-core` has 0 dependencies on `repro-mlflow`, `deepscratch`, or `studies/`. |
| **`studies/`** | Domain-specific reproduction studies and experimental protocols (`studies/dlfs`, `studies/f2`). | Orchestrates `packages/` engines and compares against `references/`. Contains explicit **adapters**, study catalogs, configs, and custom analysis scripts. |
| **`references/`** | Vendored immutable upstream baselines (`references/dlfs1-book`, `references/dlfs2-book`). | Read-only snapshots of upstream code. Must include `provenance.json`. |
| **External services** | Managed MLflow and PostgreSQL deployments. | Durable tracking and transactional storage; see the storage contract. |
| **`artifacts/`** | Human-readable final analysis reports, paper figures, and markdown summaries. | Ephemeral scratch data must NOT be committed here. |
| **`data/`** | Canonical dataset directory (`data/mnist`, `data/ptb`, `data/sequence`). | Managed via 3-tier fallback resolution. Ignored by Git. |
| **`.staging/`** | Volatile in-memory / scratch buffer during live runs. | Completely disposable (`rm -rf .staging` safe). |
| **`.cache/`** | Derived local cache (MLflow artifact downloads, analysis intermediate caches). | 100% reconstructible (`rm -rf .cache` safe). |

---

## 3. Core Architectural Invariants

1. **Unidirectional Dependency Hierarchy:**
   $$\text{studies} \longrightarrow \text{repro-mlflow} \longrightarrow \text{repro-core} \longrightarrow \text{external third-party (numpy/psutil)}$$
   No circular dependencies across packages or from packages back to studies are permitted.
2. **Zero-Dependency Engine Guarantee:**
   `packages/deepscratch` is a 100% standalone deep learning engine with **zero imports** from `repro_core`, `repro_mlflow`, or `dlfs`.
3. **Explicit Translation Adapter Boundary:**
   Translation between repository representations (`RunPlan`, `RuntimePaths`, YAML configs, `CheckpointManager`) and engine-native objects occurs exclusively in study adapters ([`docs/architecture/adapters.md`](file:///home/esillileu/abstract-dissection/docs/architecture/adapters.md)).
4. **Decoupled Checkpoint Contract:**
   `repro-core` owns checkpoint retention policies, generational naming, atomic staging, and pointer JSONs (`latest.json`, `best.json`, `final.json`), with **zero coupling** to deep learning parameters or serialization formats. Study adapters own state serialization and restoration.
5. **Reproducibility Contract:**
   All executions must declare exact seeds, backend targets (CPU/GPU), precision dtypes, and configuration digests.
6. **Module-Scoped Study Isolation:**
   `repro-core` dispatches execution dynamically to study adapters through declared `executor_module`s (`ExecutionDefinition.executor_module`). No process-global registries or cross-study name collisions exist.
