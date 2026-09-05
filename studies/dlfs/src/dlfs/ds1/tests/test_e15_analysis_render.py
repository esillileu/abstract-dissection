from dlfs.ds1.analysis import render


def test_e15_loads_reused_e07_deep_cnn_runs() -> None:
    assert render.STUDY_SOURCES["e15"] == ("e07", "e15")
