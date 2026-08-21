from __future__ import annotations

from exp.deepscratch.ds2.profile.e02.validation import run


def test_e02_fused_paths_match_dense_lockstep_on_cpu(tmp_path) -> None:
    result = run(tmp_path / "e02_fused_validation.json")

    assert result["passed"] is True
    assert result["comparisons"]["cpu"]["cbow"]["passed"] is True
    assert result["comparisons"]["cpu"]["skipgram"]["passed"] is True
