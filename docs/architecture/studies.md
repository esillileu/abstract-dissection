# Studies Architecture & Plugin Interface

The `studies/` directory contains reproduction studies, experimental campaigns, benchmark suites, and publication analyses.

---

## 1. Role of Studies vs. Packages

* **`packages/` provide *Mechanisms*:** Reusable math libraries, neural network layers, execution protocols, tracking clients.
* **`studies/` define *Policies, Hypotheses & Orchestration*:** Specific experiment catalogs, dataset pairings, comparison baselines, observation runners, and domain-specific figures.
* **Study Adapters bridge *Representations*:** Translate study YAML configurations and runtime resources into engine-native models, losses, optimizers, and checkpoint handlers.

---

## 2. Study Structure (e.g. `studies/dlfs`)

```text
studies/dlfs/
├── pyproject.toml                     # Declares repro-core, repro-mlflow, deepscratch as workspace deps
└── src/dlfs/
    ├── plugin.py                      # Repro CLI Plugin Entrypoint (discoverable by repro_core)
    ├── cli.py                         # Study-specific subcommands (plan, run, analyze, status)
    ├── events.py                      # Study-level event recorders and metric handlers
    ├── adapters/                      # Study-wide representation adapters
    │   └── checkpoint.py              # DeepScratch state serialization & CheckpointManager factory
    ├── ds1/                           # Deep Learning From Scratch Vol 1 (MNIST / CNN / MLP)
    │   ├── config/                    # YAML run specifications (implemented & original)
    │   ├── implemented/
    │   │   ├── adapters/              # DS1 representation adapters (models, objectives, optimizers, data)
    │   │   ├── executor.py            # DS1 experiment orchestration & evaluation scheduling
    │   │   ├── final_gap.py           # Final checkpoint parity metrics evaluator
    │   │   └── backfill_full_train_gap.py
    │   ├── original/                  # Upstream reference executor (using references/dlfs1-book)
    │   ├── analysis/                  # Comparison curves, loss trajectories, filter visualizations
    │   └── tests/                     # DS1 unit, adapter, and smoke tests
    ├── ds2/                           # Deep Learning From Scratch Vol 2 (NLP / Word2Vec / RNN / Seq2Seq)
    │   ├── config/                    # YAML run specifications (implemented & original)
    │   ├── implemented/
    │   │   ├── adapters/              # DS2 representation adapters (models, objectives, optimizers, batch, data)
    │   │   ├── executor.py            # DS2 experiment orchestration & evaluation scheduling
    │   │   └── observations.py        # Sequence and attention observation runners
    │   ├── original/                  # Upstream reference executor (using references/dlfs2-book)
    │   ├── profile/                   # Performance profiling and execution breakdown studies
    │   ├── analysis/                  # Perplexity curves, word vector alignments, attention maps
    │   └── tests/                     # DS2 unit, adapter, and smoke tests
    └── analysis/                      # Cross-volume synthesis & report renderers
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

`repro-core` discovers studies dynamically at runtime through a standardized plugin contract in [`studies/*/src/*/plugin.py`](file:///home/esillileu/abstract-dissection/studies/dlfs/src/dlfs/plugin.py):

```python
# plugin.py
class StudyPlugin:
    @property
    def name(self) -> str:
        """Name of the study as invoked via CLI (e.g., 'dlfs')."""
        return "dlfs"

    def register_commands(self, group: click.Group) -> None:
        """Attach study-specific command groups to the central `repro` CLI."""
        group.add_command(dlfs_group)
```

This allows adding new studies (e.g., `studies/transformer`, `studies/diffusion`) without modifying a single line of code in `repro-core`.
