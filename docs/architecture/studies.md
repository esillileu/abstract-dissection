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

### 1) DLFS Reproduction Suite (`studies/dlfs` — Legacy Runner)
Reproduces the canonical "Deep Learning from Scratch" curriculum (Volumes 1 & 2):
* **`ds1/`:** Volume 1 experiments (MNIST, MLP, CNN filters, numerical gradients — e01 to e11).
* **`ds2/`:** Volume 2 experiments (Word2Vec CBOW/Skip-Gram, PTB language modeling, RNN, LSTM, Seq2Seq, Attention — e01 to e08).
* Preserved as a legacy execution system using `repro-core`'s runner. DLFS is not required to adapt to F2's single-lifecycle coordinator.

### 2) F2 Word2Vec Reproduction Campaign Suite (`studies/f2`)
A rigorous reproduction campaign for Word2Vec (Mikolov et al., 2013) following a single execution lifecycle:
* **F2-CC Producer (`studies/f2_cc`):** Independent Common Crawl candidate sampling, audit, two-phase estimation, and release creation backed by its own database.
* **Corpus Domain (`f2 corpus`):** Acquisition, normalization, and validation of completed releases consumed by downstream experiments.
* **Reproduction Suites (`f2/suites/`):**
  * `w2v1`: Efficient Estimation of Word Representations in Vector Space (CBOW, Skip-Gram, NNLM).
  * `w2v2`: Distributed Representations of Words and Phrases and their Compositionality (Word/Phrase Skip-Gram, NEG/HS/NCE, subsampling).
  * `suites/common/w2v`: Reusable W2V algorithmic preparation, training coordination, checkpoint codec, and analogy evaluation.

```text
studies/
├── dlfs/                              # DLFS Vol 1 & Vol 2 Reproduction Suite (Legacy)
│   ├── pyproject.toml
│   └── src/dlfs/
│       ├── ds1/                       # Volume 1: Vision & Feedforward
│       ├── ds2/                       # Volume 2: NLP & Sequence Modeling
│       └── adapters/
├── f2_cc/                             # Independent Common Crawl producer
│   └── src/f2_cc/
│       ├── corpus/                    # Sampling, audit, analysis, release creation
│       └── db/                        # Dedicated f2_cc operational schema
└── f2/                                # F2 Word2Vec Reproduction Campaign
    ├── pyproject.toml
    ├── catalog/                       # Static reproduction catalog manifest and offline validator
    │   ├── w2v.json                   # W2V1/W2V2 catalog input
    │   └── ...
    └── src/f2/
        ├── plugin.py                  # Plugin entrypoint (registers `repro f2 ...`)
        ├── cli/                       # Command-line subcommands (plan, run, check, analyze, evaluate, data)
        ├── execution/                 # F2 single execution coordinator & condition identity
        ├── storage/                   # F2 run persistence, MLflow publication, analysis cache
        ├── common/                    # Shared statistics, analysis helpers, paths
        ├── corpus/                    # Completed-source preparation and validation
        ├── catalog/                   # Static catalog manifest parsing and validation
        └── suites/                    # Paper suites (w2v1, w2v2) & common/w2v
```

---

## 3. Separation of Responsibilities: Adapters vs. Executors

`studies/f2_cc` is an independent corpus-producer study backed by its own
`f2_cc` database. It owns Common Crawl discovery, range acquisition, extraction,
sampling, audit, and feasibility analysis. `studies/f2` owns catalog management
and preprocessing of completed corpus releases only.

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

### Research runs are not test harnesses

A selected study slot maps to one research run that executes the declared
scientific schedule to completion. Production suite catalogs and executors must
not contain smoke variants, fixture-backed variants, deliberate interruption,
resume-parity exercises, shortened test schedules, or extra MLflow attempts whose
purpose is implementation verification. Such behavior belongs exclusively in
test code and test-owned fixtures.

Runtime integrity enforcement is distinct from test behavior. Dataset and
manifest identity checks, deterministic configuration, checkpoint writing, and
artifact-upload confirmation are part of the reproducibility contract when they
observe or protect the declared run without changing its schedule or splitting
it into synthetic attempts.



F2 keeps extraction options, text filters/counters, sampling, auditing, estimation,
normalization, record-preserving sharding and manifests because these determine
research inputs and measurements. `corpus/storage.py` owns DOC serialization and
provenance export. Corpus DBML, repository and migration tree remain in F2 (the
legacy catalog DB has been removed; experiment planning and execution are 100% YAML-driven).
F2 owns only these research-domain schemas and migrations; the PostgreSQL instance,
roles, backups, upgrades, and availability remain external responsibilities.
Small session/migration helpers do not justify another generic package. The actual
feasibility estimator is in corpus/analysis.py; common/stats contains supporting
research calculations. No Rust engine interface is predeclared by this refactor.

F2 W2V1/W2V2 execution and analysis follow the [five-layer experiment lifecycle](f2-experiments.md). Execution is YAML-driven; execution completion belongs exclusively to MLflow.
