# Studies Architecture & Plugin Interface

The `studies/` directory contains reproduction studies, experimental campaigns, benchmark suites, and publication analyses.

---

## 1. Role of Studies vs. Packages

* **`packages/` provide *Mechanisms*:** Reusable math libraries, neural network layers, execution protocols, tracking clients.
* **`studies/` define *Policies & Hypotheses*:** Specific experiment catalogs, dataset pairings, comparison baselines, and domain-specific figures.

---

## 2. Study Structure (e.g. `studies/dlfs`)

```text
studies/dlfs/
├── pyproject.toml                     # Declares repro-core, repro-mlflow, deepscratch as workspace deps
└── src/dlfs/
    ├── plugin.py                      # Repro CLI Plugin Entrypoint (discoverable by repro_core)
    ├── cli.py                         # Study-specific subcommands (plan, run, analyze, status)
    ├── events.py                      # Study-level event recorders and metric handlers
    ├── ds1/                           # Deep Learning From Scratch Vol 1 (MNIST / CNN / MLP)
    │   ├── config/                    # YAML run specifications (implemented & original)
    │   ├── implemented/               # Custom engine execution adapter (using deepscratch)
    │   ├── original/                  # Upstream reference executor (using references/dlfs1-book)
    │   ├── analysis/                  # Comparison curves, loss trajectories, filter visualizations
    │   └── tests/                     # DS1 unit and smoke tests
    ├── ds2/                           # Deep Learning From Scratch Vol 2 (NLP / Word2Vec / RNN / Seq2Seq)
    │   ├── config/                    # YAML run specifications (implemented & original)
    │   ├── implemented/               # Custom engine execution adapter (using deepscratch)
    │   ├── original/                  # Upstream reference executor (using references/dlfs2-book)
    │   ├── profile/                   # Performance profiling and execution breakdown studies
    │   ├── analysis/                  # Perplexity curves, word vector alignments, attention maps
    │   └── tests/                     # DS2 unit and smoke tests
    └── analysis/                      # Cross-volume synthesis & report renderers
```

---

## 3. Dynamic Plugin Discovery Interface

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
