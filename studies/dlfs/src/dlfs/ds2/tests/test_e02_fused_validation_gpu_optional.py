from __future__ import annotations

import pytest

from dlfs.ds2.profile.e02.validation import run

cp = pytest.importorskip("cupy")


def test_e02_fused_paths_match_dense_lockstep_on_cuda(tmp_path) -> None:
    try:
        result = run(
            tmp_path / "e02_fused_validation.json",
            devices=("cuda:0",),
        )
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA runtime unavailable: {error}")

    assert result["passed"] is True
    assert result["comparisons"]["cuda:0"]["cbow"]["passed"] is True
    assert result["comparisons"]["cuda:0"]["skipgram"]["passed"] is True
