"""Statistical contracts for shared-question corpus comparisons."""

import numpy as np

from f2.suites.w2v1.corpus_comparison import correlation, percentile_interval, resamples


def test_controls_remove_spurious_frequency_association():
    rng = np.random.default_rng(12)
    frequency = rng.normal(size=1000)
    x = (frequency + rng.normal(scale=0.05, size=1000))[:, None]
    y = (frequency + rng.normal(scale=0.05, size=1000))[:, None]
    unadjusted = correlation(x, y, np.ones((1000, 1)), np.ones(1000))
    adjusted = correlation(
        x, y, np.column_stack((np.ones(1000), frequency)), np.ones(1000)
    )
    assert unadjusted[0, 0] > 0.99
    assert abs(adjusted[0, 0]) < 0.1


def test_paired_resampling_keeps_shared_sources_and_seeds_identical():
    rows = [
        dict(category="fixture", word_a="c", word_b="d"),
        dict(category="fixture", word_a="e", word_b="f"),
    ]
    questions = {
        i: dict(
            category="fixture",
            words=words,
            scores=np.array([[0, 100, 0], [100, 0, 100]], dtype=float)
            if words[:2] == ("a", "b")
            else np.array([[100, 0, 100], [0, 100, 0]], dtype=float),
        )
        for i, words in enumerate(
            (
                ("a", "b", "c", "d"),
                ("g", "h", "c", "d"),
                ("a", "b", "e", "f"),
                ("g", "h", "e", "f"),
            )
        )
    }
    first = list(resamples(rows, questions, np.random.default_rng(42), 20))
    second = list(resamples(rows, questions, np.random.default_rng(42), 20))
    np.testing.assert_allclose(first[0][0][:, :2], 50)
    for (a, aw), (b, bw) in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(aw, bw)
        np.testing.assert_allclose(a[:, 2], a[:, 1] - a[:, 0])
        # Shared source scores stay identical despite distinct target multipliers.
        np.testing.assert_allclose(a[0], a[1])
    assert np.std([w[0] for _, w in first[1:]]) > 0
    assert np.std([a[0, 0] for a, _ in first[1:]]) > 0


def test_unidentified_feature_interval_stays_missing():
    values = np.array([[[np.nan, 0.1]], [[np.nan, 0.2]], [[np.nan, 0.3]]])
    interval = percentile_interval(values)
    assert np.isnan(interval[:, 0, 0]).all()
    np.testing.assert_allclose(interval[:, 0, 1], [0.105, 0.295])
