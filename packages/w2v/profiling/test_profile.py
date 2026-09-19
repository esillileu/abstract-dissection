import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "w2v_profile", Path(__file__).with_name("profile.py")
)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def test_runner_metrics_parses_valid_output() -> None:
    result = PROFILE.runner_metrics(
        "training_seconds=1.25 processed_tokens=500 processed_tokens_per_training_second=400 vocab_size=8",
        "rust",
    )
    assert result == {
        "training_seconds": 1.25,
        "processed_tokens": 500,
        "training_tokens_per_second": 400.0,
    }


def test_runner_metrics_rejects_missing_value() -> None:
    with pytest.raises(RuntimeError, match="missing"):
        PROFILE.runner_metrics(
            "training_seconds=1.25 processed_tokens=500", "reference"
        )


@pytest.mark.parametrize(
    "output",
    [
        "training_seconds=0 processed_tokens=500 processed_tokens_per_training_second=400",
        "training_seconds=NaN processed_tokens=500 processed_tokens_per_training_second=400",
        "training_seconds=1.25 processed_tokens=-1 processed_tokens_per_training_second=400",
    ],
)
def test_runner_metrics_rejects_non_positive_or_non_finite(output: str) -> None:
    with pytest.raises(RuntimeError, match="invalid"):
        PROFILE.runner_metrics(output, "rust")


def test_original_has_no_training_only_metrics() -> None:
    assert PROFILE.runner_metrics("anything", "original") is None
