import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from experiments.deepscratch1.analysis.common import DEFAULT_ERROR_BARS, ErrorBarStyle, curve_from_histories, smooth_curve


def test_curve_uses_mean_and_min_max_for_each_available_log_step() -> None:
    curve = curve_from_histories([{0: 1.0, 2: 5.0}, {0: 3.0, 1: 7.0, 2: 9.0}])

    assert curve.run_count == 2
    np.testing.assert_array_equal(curve.steps, [0, 1, 2])
    np.testing.assert_allclose(curve.mean, [2.0, 7.0, 7.0])
    np.testing.assert_allclose(curve.minimum, [1.0, 7.0, 5.0])
    np.testing.assert_allclose(curve.maximum, [3.0, 7.0, 9.0])


def test_error_bar_style_has_sparse_global_default_and_per_analysis_override() -> None:
    assert DEFAULT_ERROR_BARS.every == 5
    assert ErrorBarStyle(every=2).every == 2


def test_smooth_curve_matches_the_book_window_shape() -> None:
    values = np.arange(20, dtype=float)

    result = smooth_curve(values)

    assert result.shape == values.shape
    np.testing.assert_allclose(result[5:15], values[5:15])
