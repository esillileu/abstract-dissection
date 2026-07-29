from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from exp.analyze import RunRef
from exp.model_parameters import (
    ParameterCount,
    append_parameter_counts,
    count_model_parameters,
    count_parameter_manifest,
    parameter_count_for_runs,
)


class FakeClient:
    def download_artifacts(self, run_id, artifact_path):
        raise FileNotFoundError((run_id, artifact_path))


def _run(tmp_path, name: str, count: int) -> RunRef:
    root = tmp_path / name
    path = root / "model"
    path.mkdir(parents=True)
    (path / "parameter_manifest.json").write_text(
        json.dumps(
            [
                {"name": "weight", "numel": count - 2},
                {"name": "bias", "numel": 2},
            ]
        ),
        encoding="utf-8",
    )
    return RunRef(name, "MODEL", name, 0, root)


def test_count_model_parameters_counts_shared_parameter_once():
    shared = SimpleNamespace(data=np.zeros((2, 3)))
    bias = SimpleNamespace(data=np.zeros(4))
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            [("left.weight", shared), ("right.weight", shared), ("bias", bias)]
        )
    )

    assert count_model_parameters(model) == 10


def test_count_parameter_manifest_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate model parameter name"):
        count_parameter_manifest(
            [
                {"name": "weight", "numel": 6},
                {"name": "weight", "numel": 6},
            ]
        )


def test_parameter_count_for_runs_requires_same_model_size(tmp_path):
    client = FakeClient()
    count = parameter_count_for_runs(
        client,
        [_run(tmp_path, "seed-1", 10), _run(tmp_path, "seed-2", 10)],
    )

    assert count is not None
    assert count.value == 10
    assert count.run_count == 2

    with pytest.raises(ValueError, match="differ across seed runs"):
        parameter_count_for_runs(
            client,
            [_run(tmp_path, "seed-3", 11), _run(tmp_path, "seed-4", 12)],
        )


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (
            "series,seed_runs,unit,mean,standard_deviation,minimum,maximum\n",
            "MODEL/parameter_count,2,parameters,10,,,\n",
        ),
        (
            "series,metric,seed_runs,unit,mean,standard_deviation,minimum,maximum\n",
            "MODEL,parameter_count,2,parameters,10,,,\n",
        ),
    ],
)
def test_append_parameter_counts_supports_both_summary_schemas(
    tmp_path,
    header,
    expected,
):
    output = tmp_path / "summary.csv"
    output.write_text(header, encoding="utf-8")

    append_parameter_counts(
        output,
        {"MODEL": ParameterCount(value=10, run_count=2)},
    )

    assert output.read_text(encoding="utf-8") == header + expected
