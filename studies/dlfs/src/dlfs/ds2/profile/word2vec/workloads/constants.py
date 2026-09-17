from __future__ import annotations

from dlfs.ds2.profile.paths import REPOSITORY_ROOT, profile_measurements
from repro_core.context.paths import RuntimePaths

ROOT = REPOSITORY_ROOT
BOOK_ROOT = ROOT / str(
    RuntimePaths.from_environment().reference("dlfs2-book") / "source"
)
PTB_TRAIN = RuntimePaths.from_environment().dataset("ptb") / "ptb.train.npy"
DEFAULT_OUTPUT = profile_measurements("e10") / "update.json"
DEFAULT_EPOCHS = 10

CONDITIONS = (
    "original-cbow-onehot-fs",
    "original-cbow-fs",
    "original-cbow-ns",
    "original-skipgram-onehot-fs",
    "original-skipgram-fs",
    "original-skipgram-ns",
    "implemented-cbow-onehot-fs",
    "implemented-cbow-fs",
    "implemented-cbow-ns",
    "implemented-cbow-fused-ns",
    "implemented-skipgram-onehot-fs",
    "implemented-skipgram-fs",
    "implemented-skipgram-ns",
    "implemented-skipgram-fused-ns",
)

STAGES = {
    # Always measures one cold update before the requested steady-state work.
    "update": {
        "warmup_updates": 5,
        "measured_updates": 10,
        "phase_updates": 0,
        "repetitions": 3,
    },
    # Enough samples to expose update-level variance without an epoch run.
    "estimate": {
        "warmup_updates": 20,
        "measured_updates": 50,
        "phase_updates": 0,
        "repetitions": 5,
    },
    # Long throughput windows plus a separately synchronized phase pass.
    "detail": {
        "warmup_updates": 50,
        "measured_updates": 200,
        "phase_updates": 5,
        "repetitions": 5,
    },
}
