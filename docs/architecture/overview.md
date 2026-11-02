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
        W2V[packages/w2v]
        IO[packages/repro-io]
    end

    subgraph Layer 3: Studies [3. Studies (Scientific Experiments & Campaigns)]
        STUDY_DLFS[studies/dlfs\nDLFS Vol 1 & 2 Suite (Legacy)]
        STUDY_F2[studies/f2\nF2 Word2Vec Reproduction Campaign]
        STUDY_F2CC[studies/f2_cc\nIndependent Common Crawl Producer]
        ADAPTERS[studies/*/adapters\nTranslation Adapters Boundary]
    end

    subgraph Layer 4: Infrastructure [4. Infra & Tracking]
        INFRA[External operators: MLflow, PostgreSQL, S3]
    end

    STUDY_DLFS -->|orchestrates via| ADAPTERS
    STUDY_F2 -->|orchestrates via| ADAPTERS
    ADAPTERS -->|uses pure retention| CORE
    ADAPTERS -->|injects datasets & builds models| ENGINE
    STUDY_F2 -->|high-performance numeric kernels| W2V
    STUDY_DLFS -->|tracks runs| MLFLOW
    STUDY_F2 -->|tracks runs| MLFLOW
    STUDY_F2CC -->|acquires bytes| IO
    STUDY_DLFS -->|reads immutable baseline| REF
    MLFLOW -->|depends on| CORE
    INFRA -.->|provides endpoints| MLFLOW
    INFRA -.->|provides database/object-store services| STUDY_F2
    INFRA -.->|provides database services| STUDY_F2CC
```

---

## 2. Directory Layout & Roles

| Top-Level Directory | Ownership / Role | Dependency Rules |
| :--- | :--- | :--- |
| **`packages/`** | Reusable, self-contained computational and infrastructure libraries (`repro-core`, `repro-mlflow`, `repro-io`, `w2v`, `deepscratch`). | **Must NOT depend on `studies/` or external references.** `deepscratch`, `w2v`, and `repro-io` have 0 dependencies on other workspace packages. `repro-core` has 0 dependencies on `repro-mlflow`, `deepscratch`, or `studies/`. |
| **`studies/`** | Domain-specific reproduction studies and experimental protocols (`studies/dlfs`, `studies/f2`, `studies/f2_cc`). | Orchestrates `packages/` engines and compares against `references/`. Contains explicit **adapters**, study catalogs, configs, and custom analysis scripts. |
| **`references/`** | Vendored immutable upstream baselines (`references/dlfs1-book`, `references/dlfs2-book`). | Read-only snapshots of upstream code. Must include `provenance.json`. |
| **External services** | MLflow, PostgreSQL, and S3-compatible services operated outside this repository. | Provide endpoints and durable infrastructure; service operation, server schemas, roles, buckets, backups, and availability are outside repository/study ownership. |
| **`workspace/`** | Unified local runtime workspace (`REPRO_WORKSPACE_ROOT`). | Contains `inputs/` (raw corpora, benchmarks, prepared data), `runs/` (local execution states, preserved runs/checkpoints), and `analysis/` (MLflow download cache, results eval cache, and final reports). |
| **Legacy paths (`data/`, `.staging/`, `.cache/`, `artifacts/`)** | Maintained for DLFS backwards compatibility only. | F2 uses the unified `workspace/` hierarchy (`inputs/`, `runs/`, `analysis/`). |

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
6. **Module-Scoped Study Isolation & Lifecycle Independence:**
   `repro-core` dispatches execution dynamically to study adapters through declared `executor_module`s (`ExecutionDefinition.executor_module`). No process-global registries or cross-study name collisions exist.
   F2 follows a dedicated single-lifecycle architecture (Coordinator + Storage + Suites) with MLflow `FINISHED` as the sole authority of execution completion. DLFS is preserved as a legacy runner system and is not forced onto the F2 execution lifecycle.

`infra/` records external service interfaces and operator handoff only; it is not
an in-repository service operations responsibility. F2 retains only research-domain
DDL, migrations, and lineage semantics.
Independent repro-io, w2v, and deepscratch sit beside the core/tracking dependency chain.
