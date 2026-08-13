from pathlib import Path

from exp.deepscratch.ds2.analysis import e01_toy_word2vec as analysis
from exp.framework.analysis.core import aggregate


def test_e01_renders_cbow_and_skipgram_separately(tmp_path: Path, monkeypatch) -> None:
    curve = aggregate([{0.0: 2.0, 1.0: 1.0}])
    monkeypatch.setattr(
        analysis,
        "runs",
        lambda _client, _group, ids: {atomic: [] for atomic in ids},
    )
    monkeypatch.setattr(analysis, "source_curve", lambda *_args: curve)

    output = tmp_path / "ds2_e01_imp.png"
    outputs = analysis.render(object(), "band", output)

    assert outputs == [
        tmp_path / "ds2_e01_imp_cbow.png",
        tmp_path / "ds2_e01_imp_skipgram.png",
        tmp_path / "ds2_e01_imp_curves.csv",
    ]
    assert all(path.is_file() for path in outputs)
    summary = outputs[-1].read_text(encoding="utf-8")
    assert "CBOW" in summary
    assert "Skip-gram" in summary
