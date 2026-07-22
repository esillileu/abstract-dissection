from pathlib import Path

import numpy as np

from mlprosection.experiment import load_yaml, normalize_config
from mlprosection.experiment.executors.supervised import _apply_input_transform, _model, _permute_pixels, _train_evaluation_probe
from mlprosection.experiment.reproducibility import configure_runtime, seed_batch_order


CONFIG = Path("experiments/deepscratch1/config/e05_batchnorm_scale.yaml")
E08_CONFIG = Path("experiments/deepscratch1/config/e08_cnn_structure.yaml")
E09_CONFIG = Path("experiments/deepscratch1/config/e09_cnn_accuracy.yaml")


def _configured_model(atomic_run_id: str):
    config = normalize_config(load_yaml(CONFIG, atomic_run_id=atomic_run_id))
    config["seed"] = 1234
    backend, streams, _ = configure_runtime(config)
    model = _model(config["model"])
    seed_batch_order(backend, streams)
    batches = backend.xp.random.randint(0, 1000, size=(3, 100))
    return model, batches


def test_e05_paired_runs_share_initial_weights_and_batch_sequence() -> None:
    off_model, off_batches = _configured_model("BN-OFF-01")
    on_model, on_batches = _configured_model("BN-ON-01")

    off_parameters = list(off_model.named_parameters())
    on_parameters = list(on_model.named_parameters())

    assert len(off_parameters) == len(on_parameters)
    for (_, off_parameter), (_, on_parameter) in zip(off_parameters, on_parameters, strict=True):
        assert off_parameter.shape == on_parameter.shape
        assert np.array_equal(off_parameter.data, on_parameter.data)
    assert np.array_equal(off_batches, on_batches)


def _e08_model(atomic_run_id: str):
    config = normalize_config(load_yaml(E08_CONFIG, atomic_run_id=atomic_run_id))
    config["seed"] = 1234
    backend, streams, _ = configure_runtime(config)
    model = _model(config["model"])
    seed_batch_order(backend, streams)
    batches = backend.xp.random.randint(0, 1000, size=(3, 100))
    return model, batches


def test_e08_paired_conditions_share_initialization_and_batch_order() -> None:
    for original_id, permuted_id in (("NN-MATCHED", "NN-MATCHED-PERMUTED"), ("CNN-SIMPLE", "CNN-SIMPLE-PERMUTED")):
        original_model, original_batches = _e08_model(original_id)
        permuted_model, permuted_batches = _e08_model(permuted_id)
        for (_, original), (_, permuted) in zip(original_model.named_parameters(), permuted_model.named_parameters(), strict=True):
            assert original.shape == permuted.shape
            assert np.array_equal(original.data, permuted.data)
        assert np.array_equal(original_batches, permuted_batches)


def test_e08_models_are_parameter_matched() -> None:
    nn_model, _ = _e08_model("NN-MATCHED")
    cnn_model, _ = _e08_model("CNN-SIMPLE")

    assert sum(parameter.data.size for _, parameter in nn_model.named_parameters()) == 433_875
    assert sum(parameter.data.size for _, parameter in cnn_model.named_parameters()) == 433_890


def test_e09_cnn_recipes_train_independently_for_twenty_epochs() -> None:
    simple = normalize_config(load_yaml(E09_CONFIG, atomic_run_id="CNN-SIMPLE-ACCURACY"))
    deep = normalize_config(load_yaml(E09_CONFIG, atomic_run_id="CNN-DEEP-ACCURACY"))

    for config in (simple, deep):
        assert config["training"]["max_epochs"] == 20
        assert config["training"]["record_step_validation_interval"] == 20
        assert config["loader"]["sampling_method"] == "with_replacement"
        assert config["execution_group_id"] == "g10"
        assert config["training"]["record_step_train_evaluation"] is True
        assert _model(config["model"])


def test_e08_and_e09_use_the_same_fixed_validation_probe() -> None:
    e08 = normalize_config(load_yaml(E08_CONFIG, atomic_run_id="CNN-SIMPLE"))
    e09_simple = normalize_config(load_yaml(E09_CONFIG, atomic_run_id="CNN-SIMPLE-ACCURACY"))
    e09_deep = normalize_config(load_yaml(E09_CONFIG, atomic_run_id="CNN-DEEP-ACCURACY"))

    for config in (e08, e09_simple, e09_deep):
        assert config["dataset"]["validation_size"] == 1_000
        assert config["dataset"]["validation_seed"] == 20260808
        assert config["training"]["record_first_validation_evaluation"] is True
        assert config["training"]["record_step_validation_interval"] == 20
    assert e08["training"]["max_epochs"] == 2
    assert e09_simple["training"]["max_epochs"] == 20
    assert e09_deep["training"]["max_epochs"] == 20


def test_step_train_evaluation_probe_is_fixed_and_opt_in(tmp_path: Path) -> None:
    class Backend:
        xp = np

    x_train = np.arange(20).reshape(10, 2)
    t_train = np.arange(10)
    disabled = _train_evaluation_probe(
        dataset={}, training={}, backend=Backend(), x_train=x_train, t_train=t_train, artifact_root=tmp_path,
    )
    first = _train_evaluation_probe(
        dataset={"train_evaluation_size": 4, "train_evaluation_seed": 42}, training={"record_step_train_evaluation": True}, backend=Backend(), x_train=x_train, t_train=t_train, artifact_root=tmp_path,
    )
    second = _train_evaluation_probe(
        dataset={"train_evaluation_size": 4, "train_evaluation_seed": 42}, training={"record_step_train_evaluation": True}, backend=Backend(), x_train=x_train, t_train=t_train, artifact_root=tmp_path,
    )

    assert disabled[0] is None
    assert disabled[2] == {"enabled": False, "size": 0}
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2]["size"] == 4
    assert (tmp_path / "data" / "train_evaluation_indices.npy").is_file()


def test_pixel_permutation_is_shape_invariant_and_artifacted(tmp_path: Path) -> None:
    class Backend:
        xp = np

    flat_train = np.arange(2 * 784).reshape(2, 784)
    flat_test = np.arange(784).reshape(1, 784)
    metadata = _apply_input_transform(
        dataset={"input_transform": {"name": "pixel_permutation", "seed": 20260808}},
        backend=Backend(),
        x_train=flat_train,
        x_test=flat_test,
        artifact_root=tmp_path,
    )
    permutation = metadata["permutation"]
    image_train = flat_train.reshape(2, 1, 28, 28)

    np.testing.assert_array_equal(_permute_pixels(flat_train, permutation), _permute_pixels(image_train, permutation).reshape(2, 784))
    assert metadata["feature_count"] == 784
    assert len(metadata["sha256"]) == 64
    assert (tmp_path / "data" / "pixel_permutation.npy").is_file()
