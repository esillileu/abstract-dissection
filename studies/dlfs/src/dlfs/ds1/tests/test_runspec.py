from __future__ import annotations

import pytest

from dlfs.ds1.implemented.spec import parse_run_spec


def test_ds1_runspec_allows_device_timing_profiling_policy() -> None:
    spec = parse_run_spec(
        "studies/dlfs/src/dlfs/ds1/config/implemented/e01_mnist_optimizer.yaml",
        atomic_run_id="MLP-OPT-SGD",
        overrides={"profiling": {"device_timing": True}},
    )

    assert spec.profiling["device_timing"] is True
    assert spec.config["profiling"] == {"enabled": False, "device_timing": True}


def test_ds1_runspec_still_rejects_legacy_training_key() -> None:
    with pytest.raises(ValueError, match="old catalog keys"):
        parse_run_spec(
            "studies/dlfs/src/dlfs/ds1/config/implemented/e01_mnist_optimizer.yaml",
            atomic_run_id="MLP-OPT-SGD",
            overrides={"training": {"max_epochs": 1}},
        )
