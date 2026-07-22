from __future__ import annotations

import pytest

from exp.ds1.spec import parse_run_spec


def test_gt03_runspec_declares_documented_offset_update_cadence() -> None:
    spec = parse_run_spec("exp/ds1/config/e03_mnist_weight_decay.yaml", atomic_run_id="REG-WD-OFF")

    trigger = spec.triggers[0]
    assert spec.identity.group_id == "GT03"
    assert trigger.type == "updates"
    assert (trigger.start, trigger.every, trigger.stop) == (1, 3, 601)
    assert trigger.sources == ("mnist-train-first-300", "mnist-test-full")


def test_gt05_runspec_declares_documented_train_probe_cadence() -> None:
    spec = parse_run_spec("exp/ds1/config/e05_mnist_batchnorm_scale.yaml", atomic_run_id="BN-SCALE-01-ON")

    trigger = spec.triggers[0]
    assert spec.identity.group_id == "GT05"
    assert (trigger.start, trigger.every, trigger.stop) == (1, 10, 191)
    assert trigger.sources == ("mnist-train-first-1000",)


def test_gt06_and_gt07_runspec_declare_fixed_first_1000_sources() -> None:
    for path, atomic_run_id, group_id in (
        ("exp/ds1/config/e06_mnist_simple_cnn.yaml", "CNN-SIMPLE-BOOK", "GT06"),
        ("exp/ds1/config/e07_mnist_deep_cnn.yaml", "CNN-DEEP-BOOK", "GT07"),
    ):
        spec = parse_run_spec(path, atomic_run_id=atomic_run_id)
        sources = {source.id: source for source in spec.evaluation_sources}

        assert spec.identity.group_id == group_id
        assert sources["mnist-train-first-1000"].count == 1000
        assert sources["mnist-test-first-1000"].count == 1000
        assert sources["mnist-test-full"].kind == "full"


def test_ds1_runspec_allows_device_timing_profiling_policy() -> None:
    spec = parse_run_spec(
        "exp/ds1/config/e01_mnist_optimizer.yaml",
        atomic_run_id="MLP-OPT-SGD",
        overrides={"profiling": {"device_timing": True}},
    )

    assert spec.profiling["device_timing"] is True
    assert spec.config["profiling"] == {"enabled": False, "device_timing": True}


def test_ds1_runspec_still_rejects_legacy_training_key() -> None:
    with pytest.raises(ValueError, match="old catalog keys"):
        parse_run_spec(
            "exp/ds1/config/e01_mnist_optimizer.yaml",
            atomic_run_id="MLP-OPT-SGD",
            overrides={"training": {"max_epochs": 1}},
        )
