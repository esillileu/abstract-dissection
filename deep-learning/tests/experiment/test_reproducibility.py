from __future__ import annotations

import numpy as np
import pytest

from mlprosection import Tensor
from mlprosection.experiment.reproducibility import configure_runtime, seed_streams
from mlprosection.datasets.spiral import load_data as load_spiral
from mlprosection.nn.layers import Dropout
from mlprosection.nn.model.architecture.word2vec import CBOW
from mlprosection.nn.sampling import UnigramSampler


def _runtime(master: int = 314159):
    return configure_runtime(
        {
            "seed": master,
            "numerics": {
                "backend": "numpy",
                "device": "cpu",
                "dtype": "float64",
                "deterministic": True,
            },
        }
    )


def _component_outputs(master: int = 314159):
    backend, streams, metadata = _runtime(master)
    model = CBOW(8, 3, backend=backend)
    batch_order = backend.random_stream("batch_order").permutation(12)

    dropout = Dropout(
        0.25,
        rng=backend.random_stream("dropout"),
    )
    dropout.forward(Tensor(backend.xp.ones((4, 5)), backend=backend))

    sampler = UnigramSampler.uniform(
        8,
        backend=backend,
        rng=backend.random_stream("negative_sampling"),
    )
    negatives = sampler.sample(
        Tensor(backend.xp.asarray([0, 1, 2]), backend=backend),
        sample_size=4,
    )
    return {
        "model_init": backend.to_numpy(model.W_in.data).copy(),
        "batch_order": backend.to_numpy(batch_order).copy(),
        "dropout": backend.to_numpy(dropout.mask).copy(),
        "negative_sampling": backend.to_numpy(negatives).copy(),
        "streams": streams,
        "metadata": metadata,
    }


def test_component_streams_reproduce_from_same_master_seed():
    first = _component_outputs()
    second = _component_outputs()

    for component in (
        "model_init",
        "batch_order",
        "dropout",
        "negative_sampling",
    ):
        np.testing.assert_array_equal(first[component], second[component])
    assert first["streams"] == second["streams"]
    assert (
        first["metadata"]["rng_policy"]
        == "independent_persistent_component_streams_v1"
    )


@pytest.mark.parametrize(
    "disturbed",
    ["model_init", "batch_order", "dropout", "negative_sampling"],
)
def test_extra_draw_from_one_stream_does_not_change_other_streams(disturbed):
    backend, _, _ = _runtime()
    baseline = {
        name: backend.random_stream(name).rand(8)
        for name in (
            "model_init",
            "batch_order",
            "dropout",
            "negative_sampling",
        )
    }

    backend, _, _ = _runtime()
    backend.random_stream(disturbed).rand(17)
    observed = {
        name: backend.random_stream(name).rand(8)
        for name in baseline
        if name != disturbed
    }

    for name, values in observed.items():
        np.testing.assert_array_equal(values, baseline[name])


def test_dataset_split_seed_is_stable_when_new_stream_is_added():
    streams = seed_streams(1984)
    legacy_children = np.random.SeedSequence(1984).spawn(4)
    legacy_dataset_seed = int(legacy_children[3].generate_state(1)[0])

    assert streams.dataset_split == legacy_dataset_seed


def test_spiral_dataset_does_not_consume_global_numpy_rng():
    np.random.seed(73)
    expected = np.random.rand(6)

    np.random.seed(73)
    load_spiral(seed=1984)
    observed = np.random.rand(6)

    np.testing.assert_array_equal(observed, expected)
