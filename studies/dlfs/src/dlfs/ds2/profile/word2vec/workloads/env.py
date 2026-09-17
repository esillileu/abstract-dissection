import importlib
import platform
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from deepscratch.core import BackendConfig, make_backend
from deepscratch.profiling import SectionRecorder

from .constants import BOOK_ROOT, PTB_TRAIN


@contextmanager
def _phase(recorder: SectionRecorder | None, name: str) -> Iterator[None]:
    if recorder is None:
        yield
    else:
        with recorder.section(name):
            yield


def _install_original_imports(device: str):
    # The book uses ``from common.np import *`` at import time.  Clear its
    # modules before switching CPU/CUDA in one default profiling process.
    for module_name in tuple(sys.modules):
        if (
            module_name == "common"
            or module_name.startswith("common.")
            or module_name == "ch04"
            or module_name.startswith("ch04.")
        ):
            sys.modules.pop(module_name, None)
    book_path = str(BOOK_ROOT)
    if book_path not in sys.path:
        sys.path.insert(0, book_path)
    config = importlib.import_module("common.config")
    use_gpu = device.startswith("cuda:")
    config.GPU = use_gpu
    xp = importlib.import_module("cupy" if use_gpu else "numpy")
    compatibility = types.ModuleType("common.np")
    compatibility.GPU = use_gpu
    compatibility.np = xp
    sys.modules["common.np"] = compatibility
    return xp


def _load_data(device: str):
    _install_original_imports(device)
    util = importlib.import_module("common.util")
    corpus = np.load(PTB_TRAIN)
    contexts, targets = util.create_contexts_target(corpus, 5)
    backend = make_backend(
        BackendConfig(
            device=device,
            dtype="float32",
            seed=1,
            profile=device.startswith("cuda:"),
        )
    )
    return backend, corpus, contexts, targets


def load_profile_data(device: str):
    """Public compatibility API for canonical profile studies."""
    return _load_data(device)


def _metadata(backend, *, stage: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "backend": backend.name,
        "device": backend.device,
        "device_name": platform.processor() or platform.machine(),
        "numpy_version": np.__version__,
        "stage": stage,
        "method": (
            "one workload-cold synchronized update; warmup; consecutive "
            "per-update CUDA events resolved with one synchronization; "
            "independent device-synchronized steady throughput windows; "
            "epoch/full-run estimate = cold + steady * (total_updates - 1); "
            "repeat standard deviations extrapolate between-window steady-rate "
            "variation linearly and treat the single cold observation as fixed; "
            "optional separately synchronized phase pass; implemented "
            "post-update loss included"
        ),
    }
    if backend.is_gpu:
        cp = backend.xp
        device_index = int(backend.device.split(":", 1)[1])
        properties = cp.cuda.runtime.getDeviceProperties(device_index)
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        metadata.update(
            {
                "device_name": device_name,
                "cupy_version": cp.__version__,
                "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            }
        )
    else:
        metadata["cupy_version"] = None
        metadata["cuda_runtime_version"] = None
    return metadata


def profile_metadata(backend, *, stage: str) -> dict[str, object]:
    """Return the stable environment metadata used by profile studies."""
    return _metadata(backend, stage=stage)
