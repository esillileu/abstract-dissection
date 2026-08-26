# Reproducibility, References & Determinism

Scientific reproduction requires exact numerical parity, deterministic random sequences, and immutable reference baselines.

---

## 1. Upstream References & Vendoring Policy

Upstream book implementations are stored in `references/` as **Vendored Hard Copies (Snapshots)** rather than Git submodules or dynamic clones.

```text
references/
├── dlfs1-book/
│   ├── provenance.json   # Upstream repository URL, commit SHA (c05f6bc...), adaptations
│   └── source/           # Immutable source code (ch01~ch08, dataset, common)
└── dlfs2-book/
    ├── provenance.json   # Upstream repository URL, commit SHA (07299f3...), adaptations
    └── source/           # Immutable source code (ch01~ch08, dataset, common)
```

### Why Vendored Snapshots?
1. **Guaranteed Long-Term Reproducibility:** External repositories may be deleted, renamed, or force-pushed. Vendored snapshots ensure the exact baseline remains accessible forever.
2. **Offline Parity:** Benchmarks and comparative experiments can run immediately upon `git clone` without network access.
3. **Full Provenance Tracking:** Every reference repository contains a `provenance.json` recording the upstream commit SHA, archive method, and necessary non-invasive adaptations (e.g. seed injection hooks).

---

## 2. Numerical Backends & Precision

`deepscratch.core.backend` abstracts hardware execution across NumPy (CPU) and CuPy (GPU):

* **CPU Target:** Pure `numpy.ndarray` operations.
* **GPU Target:** `cupy.ndarray` CUDA kernels with automatic memory pooling and device synchronization.
* **Precision Control:** Supported float types (`float64`, `float32`, `float16`).

```python
# Initializing Backend with Exact Precision and Seed
from deepscratch.core import BackendConfig, make_backend

backend = make_backend(BackendConfig(device="cpu", dtype="float64", seed=42))
```

---

## 3. Seed Streams & Independent RNG Policy

To eliminate cross-component RNG pollution (e.g. batch sampling affecting model weight initialization), `SeedStreams` generates five cryptographically isolated component streams from a single master seed:

```python
from deepscratch.core import seed_streams

streams = seed_streams(master=1)
# Provides isolated seeds for:
# - streams.master
# - streams.model_init
# - streams.batch_order
# - streams.dropout
# - streams.negative_sampling
# - streams.dataset_split
```

---

## 4. Runtime Configuration Protocol

Every experiment must invoke `configure_runtime(config)` before instantiating models or datasets:

```python
from deepscratch.core import configure_runtime

backend, streams, actual_runtime = configure_runtime(config)
# Sets PYTHONHASHSEED, numpy random seed, component stream seeds, and default backend.
```
