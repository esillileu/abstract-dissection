from __future__ import annotations

from dlfs.ds2.profile.paths import (
    profile_analysis,
    profile_artifacts,
    profile_measurements,
)


def test_profile_path_owners_are_separate_cache_coordinates() -> None:
    measurements = profile_measurements("e05")
    analysis = profile_analysis("e05")
    artifacts = profile_artifacts("e05")

    assert measurements.parts[-3:] == (
        "implemented",
        "profile",
        "measurements",
    )
    assert analysis.parts[-3:] == ("implemented", "profile", "analysis")
    assert artifacts.parts[-3:] == ("implemented", "profile", "artifacts")
    assert len({measurements, analysis, artifacts}) == 3
    assert ".cache" in measurements.parts
