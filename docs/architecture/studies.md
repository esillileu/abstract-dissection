# Studies Architecture & Plugin Interface

The `studies/` directory contains reproduction studies, experimental campaigns, benchmark suites, and publication analyses.

---

## 1. Role of Studies vs. Packages

* **`packages/` provide *Mechanisms*:** Reusable math libraries, neural network layers, execution protocols, tracking clients.
* **`studies/` define *Policies, Hypotheses & Orchestration*:** Specific experiment catalogs, dataset pairings, comparison baselines, observation runners, and domain-specific figures.
* **Study Adapters bridge *Representations*:** Translate study YAML configurations and runtime resources into engine-native models, losses, optimizers, and checkpoint handlers.

---

## 2. Study Structure & Catalogs

Studies are grouped into cohesive experimental volumes or research campaigns. Each study suite exposes its own subcommands through the `repro` CLI plugin system:

### 1) DLFS Reproduction Suite (`studies/dlfs`)
Reproduces the canonical "Deep Learning from Scratch" curriculum (Volumes 1 & 2):
* **`ds1/`:** Volume 1 experiments (MNIST, MLP, CNN filters, numerical gradients — e01 to e11).
* **`ds2/`:** Volume 2 experiments (Word2Vec CBOW/Skip-Gram, PTB language modeling, RNN, LSTM, Seq2Seq, Attention — e01 to e08).

### 2) F2 Research Campaign Suite (`studies/f2`)
A comprehensive reproduction and analysis campaign comprising **8~10 distinct sub-studies**:
* **Corpus Pipeline (`f2 corpus` / Pre-requisite Phase 0):** Common Crawl web-scale sampling, auditing, two-phase difference estimation, and corpus extraction for Word2Vec 33B / 1B pretraining.
* **Corpus-Dependent Studies (2 Studies):** Word2Vec pretraining dynamics, vocabulary scaling, and representation evaluations trained directly on the constructed corpus.
* **Independent Studies (5 Studies):** Dedicated theoretical, architecture, and embedding benchmark studies.

```text
studies/
├── dlfs/                              # DLFS Vol 1 & Vol 2 Reproduction Suite
│   ├── pyproject.toml
│   └── src/dlfs/
│       ├── ds1/                       # Volume 1: Vision & Feedforward
│       ├── ds2/                       # Volume 2: NLP & Sequence Modeling
│       └── adapters/
└── f2/                                # F2 Research & Benchmark Campaign
    ├── pyproject.toml
    └── src/f2/
        ├── plugin.py                  # Plugin entrypoint (registers `repro f2 ...`)
        ├── cli.py                     # Root CLI dispatcher (`repro f2 corpus ...`, `repro f2 catalog ...`)
        ├── definition.py              # F2 ExecutionDefinitions registry
        ├── common/                    # Shared network, storage, statistics, analysis & adapters
        │   ├── paths.py               # RuntimePaths centralized helper
        │   ├── network/               # TokenBucketLimiter, RangeFetcher
        │   ├── storage/               # TableExporter (Parquet/JSONL), CleanTextWriter
        │   ├── stats/                 # BootstrapVarianceEngine, DifferenceEstimator, ClassifierMetrics
        │   ├── analysis/              # Theme, declarations, BaseAnalysisOrchestrator
        │   └── adapters/              # CheckpointAdapter
        ├── corpus/                    # Common Crawl extraction pipeline & operational DB
        │   ├── cdx.py, discovery.py, pipeline.py, fetcher.py, storage.py, analysis.py, calibration.py
        │   ├── cli.py                 # `repro f2 corpus ...`
        │   └── db/                    # Corpus operational state PostgreSQL DB
        ├── catalog/                   # Reproduction catalog, resource tracking & execution plans
        │   ├── schema.dbml            # DBML architectural schema specification
        │   ├── materializer.py        # Planned run slot materializer from repro-core Planner
        │   ├── cli.py                 # `repro f2 catalog ...`
        │   └── db/                    # Catalog PostgreSQL repository & migrations
        └── suites/                    # Downstream experimental volumes (w2v_pretrain, vocab, bench)
```

---

## 3. Separation of Responsibilities: Adapters vs. Executors

| Layer | Location | Responsibilities | Forbidden Patterns |
| :--- | :--- | :--- | :--- |
| **Translation Adapters** | `studies/*/adapters/`, `studies/*/implemented/adapters/` | * Construct engine models, loss functions, optimizers, batch adapters.<br>* Inject `RuntimePaths` into dataset loaders.<br>* Serialize/restore engine checkpoint states. | ❌ Must NOT contain training loops, evaluation scheduling, metric calculations, or plotting logic. |
| **Study Executors** | `studies/*/implemented/executor.py` | * Orchestrate training loops and iteration budgets.<br>* Manage evaluation frequency, test splits, and metric recording.<br>* Handle learning rate decays and observation hooks. | ❌ Must NOT hardcode dataset paths or raw state serialization logic. |

See [`docs/architecture/adapters.md`](file:///home/esillileu/abstract-dissection/docs/architecture/adapters.md) for detailed adapter contracts and examples.

---

## 4. Dynamic Plugin Discovery Interface

`repro-core` discovers studies dynamically at runtime through a standardized plugin contract declared as an entry point in `studies/*/pyproject.toml`:

```toml
[project.entry-points."repro.studies"]
dlfs = "dlfs.plugin:PLUGIN"
```

The plugin entrypoint implements the `StudyPlugin` protocol:

```python
# plugin.py
@dataclass(frozen=True)
class DLFSStudyPlugin:
    name: str = "dlfs"

    def register_commands(self, groups: CommandGroups) -> None:
        """Attach study-specific command groups to the central `repro` CLI."""
        from . import cli

        groups.plan.command(self.name, help="Inspect run plans.")(cli.plan)
        groups.run.command(self.name, help="Execute experiments.")(cli.run)
        groups.analyze.command(self.name, help="Render results.")(cli.analyze)
        groups.check.command(self.name, help="Compare recorded run state.")(cli.check)
        groups.profile.command(self.name, help="Profile runtimes.")(cli.profile)

        # Also register study-first CLI subcommand group: repro dlfs ...
        dlfs_app = typer.Typer(name=self.name, no_args_is_help=True)
        ...
        groups.root.add_typer(dlfs_app, name=self.name)
```

Plugin discovery reports visible diagnostic warnings if a third-party study entry point fails to load, while preserving full functionality for healthy studies.

---

## 5. Module-Scoped Executor Resolution & Isolation

To support multiple independent studies in the monorepo (e.g. `dlfs`, `f2`, `transformer`) without naming collisions:

1. **No Process-Global Registry:** Experiment kinds (such as `"word2vec"` or `"supervised_classification"`) are **not** registered in a global singleton registry.
2. **Explicit Module Scoping:** Each study catalog declares its execution adapter module via `ExecutionDefinition(executor_module="...")`.
3. **Module Dispatch Contract:** The study's executor module provides a `get_executor(kind: str)` function or `EXECUTORS` mapping:

```python
# In studies/f2/src/f2/executor.py
_EXECUTORS = {
    "word2vec": F2Word2VecExecutor(),
}


def get_executor(kind: str):
    try:
        return _EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown F2 experiment kind: {kind}") from exc
```

When `run_config` or `run_yaml` executes, it dynamically dispatches to the specified `executor_module`, ensuring complete isolation across studies.

