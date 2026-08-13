from __future__ import annotations

import pytest

from exp.deepscratch.ds1.implemented.spec import parse_run_spec


def test_gt03_runspec_declares_documented_offset_update_cadence() -> None:
    spec = parse_run_spec("exp/deepscratch/ds1/config/implemented/e03_mnist_weight_decay.yaml", atomic_run_id="REG-WD-OFF")

    trigger = spec.triggers[0]
    assert spec.identity.group_id == "GT03"
    assert trigger.type == "updates"
    assert (trigger.start, trigger.every, trigger.stop) == (1, 3, 601)
    assert trigger.sources == ("mnist-train-first-300", "mnist-test-full")


def test_gt05_runspec_declares_documented_train_probe_cadence() -> None:
    spec = parse_run_spec("exp/deepscratch/ds1/config/implemented/e05_mnist_batchnorm_scale.yaml", atomic_run_id="BN-SCALE-01-ON")

    trigger = spec.triggers[0]
    assert spec.identity.group_id == "GT05"
    assert (trigger.start, trigger.every, trigger.stop) == (1, 10, 191)
    assert trigger.sources == ("mnist-train-first-1000",)


def test_gt06_and_gt07_runspec_declare_fixed_first_1000_sources() -> None:
    for path, atomic_run_id, group_id in (
        ("exp/deepscratch/ds1/config/implemented/e06_mnist_simple_cnn.yaml", "CNN-SIMPLE-BOOK", "GT06"),
        ("exp/deepscratch/ds1/config/implemented/e07_mnist_deep_cnn.yaml", "CNN-DEEP-BOOK", "GT07"),
    ):
        spec = parse_run_spec(path, atomic_run_id=atomic_run_id)
        sources = {source.id: source for source in spec.evaluation_sources}

        assert spec.identity.group_id == group_id
        assert sources["mnist-train-first-1000"].count == 1000
        assert sources["mnist-test-first-1000"].count == 1000
        assert sources["mnist-test-full"].kind == "full"


def test_gt08_runspec_declares_20_update_train_and_test_cadence() -> None:
    spec = parse_run_spec("exp/deepscratch/ds1/config/implemented/e08_mnist_spatial_layout.yaml", atomic_run_id="NN-MATCHED")
    sources = {source.id: source for source in spec.evaluation_sources}

    assert spec.identity.group_id == "GT08"
    assert sources["mnist-train-first-1000"].split == "train"
    assert sources["mnist-train-first-1000"].kind == "first_n"
    assert sources["mnist-train-first-1000"].count == 1000
    assert sources["mnist-test-first-1000"].split == "test"
    assert sources["mnist-test-first-1000"].kind == "first_n"
    assert sources["mnist-test-first-1000"].count == 1000
    assert sources["mnist-test-full"].split == "test"
    assert sources["mnist-test-full"].kind == "full"
    assert spec.triggers[0].type == "updates"
    assert (spec.triggers[0].start, spec.triggers[0].every, spec.triggers[0].stop) == (20, 20, None)
    assert spec.triggers[0].sources == ("mnist-train-first-1000", "mnist-test-first-1000")
    assert spec.triggers[1].type == "epoch_end"
    assert spec.triggers[1].sources == ("mnist-test-full",)


def test_gt09_extended_mlp_reuses_deepcnn_training_protocol() -> None:
    mlp = parse_run_spec(
        "exp/deepscratch/ds1/config/implemented/e12_mnist_extended_mlp.yaml",
        atomic_run_id="MLP-EXT-ALL-BOOK",
    )
    cnn = parse_run_spec(
        "exp/deepscratch/ds1/config/implemented/e07_mnist_deep_cnn.yaml",
        atomic_run_id="CNN-DEEP-BOOK",
    )

    assert mlp.identity.group_id == "GT09"
    assert mlp.model == {
        "name": "MLP",
        "input_size": 784,
        "hidden_sizes": [100, 100, 100, 100, 100, 100],
        "output_size": 10,
        "activation": "relu",
        "initializer": "he",
        "use_batchnorm": True,
        "dropout_ratio": 0.2,
    }
    assert mlp.optimizer == {
        "name": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.1,
    }
    assert mlp.dataset["flatten"] is True
    assert cnn.dataset["flatten"] is False
    for field in ("train_limit", "test_limit"):
        assert mlp.dataset[field] == cnn.dataset[field]
    assert mlp.loader == cnn.loader
    assert mlp.budget == cnn.budget
    assert mlp.seed_policy == cnn.seed_policy
    assert mlp.evaluation_sources == cnn.evaluation_sources
    assert mlp.triggers == cnn.triggers
    assert mlp.numerics == cnn.numerics


def test_go01_runspec_declares_optimizer_specific_learning_rates() -> None:
    expected = {
        "TOY-SGD": ("toy_sgd", 0.95),
        "TOY-MOMENTUM": ("toy_momentum", 0.1),
        "TOY-ADAGRAD": ("toy_adagrad", 1.5),
        "TOY-ADAM": ("toy_adam", 0.3),
    }

    for atomic_run_id, (name, learning_rate) in expected.items():
        spec = parse_run_spec(
            "exp/deepscratch/ds1/config/implemented/e09_optimizer_trajectory.yaml",
            atomic_run_id=atomic_run_id,
        )

        assert spec.identity.group_id == "GO01"
        assert spec.optimizer["name"] == name
        assert spec.optimizer["learning_rate"] == learning_rate


def test_ds1_runspec_allows_device_timing_profiling_policy() -> None:
    spec = parse_run_spec(
        "exp/deepscratch/ds1/config/implemented/e01_mnist_optimizer.yaml",
        atomic_run_id="MLP-OPT-SGD",
        overrides={"profiling": {"device_timing": True}},
    )

    assert spec.profiling["device_timing"] is True
    assert spec.config["profiling"] == {"enabled": False, "device_timing": True}


def test_ds1_runspec_still_rejects_legacy_training_key() -> None:
    with pytest.raises(ValueError, match="old catalog keys"):
        parse_run_spec(
            "exp/deepscratch/ds1/config/implemented/e01_mnist_optimizer.yaml",
            atomic_run_id="MLP-OPT-SGD",
            overrides={"training": {"max_epochs": 1}},
        )
