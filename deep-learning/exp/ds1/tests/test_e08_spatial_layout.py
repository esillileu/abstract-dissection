from __future__ import annotations

import numpy as np

from exp.ds1.analyze.e08_spatial_layout import _permute_image


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
