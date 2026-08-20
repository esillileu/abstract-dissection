from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from exp.deepscratch.ds1.analysis.e08_spatial_layout import (
    _add_permutation_examples,
    _permute_image,
)


def test_permute_image_matches_the_executor_flat_pixel_transform() -> None:
    image = np.arange(12).reshape(1, 3, 4)
    seed = 20260808
    expected_permutation = np.random.default_rng(seed).permutation(image.size)

    permuted = _permute_image(image, seed=seed)

    np.testing.assert_array_equal(
        permuted.reshape(-1),
        image.reshape(-1)[expected_permutation],
    )
    assert permuted.shape == image.shape


def test_permutation_examples_have_no_titles_and_are_stacked_vertically() -> None:
    figure, axis = plt.subplots()
    image = np.arange(16).reshape(1, 4, 4)

    _add_permutation_examples(axis, image, image[::-1])

    image_axes = axis.child_axes
    assert [image_axis.get_title() for image_axis in image_axes] == ["", ""]
    assert image_axes[0].get_position().x0 == image_axes[1].get_position().x0
    assert image_axes[0].get_position().y0 > image_axes[1].get_position().y0
    plt.close(figure)
