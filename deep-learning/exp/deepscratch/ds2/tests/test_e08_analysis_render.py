import json
from pathlib import Path

import numpy as np

from exp.deepscratch.ds2.analysis import e08_attention as analysis


def test_implemented_render_metadata_is_indexed_by_example_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metadata = tmp_path / "attention_render.json"
    metadata.write_text(
        json.dumps(
            {
                "input_reversal": True,
                "examples": [
                    {
                        "example_id": 4316,
                        "source_labels": ["I", "R", "F"],
                        "target_labels": ["1", "9", "8"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(analysis, "artifact_file", lambda *_args: metadata)

    labels = analysis._labels(object(), [object()])

    assert labels["4316"]["source_labels"] == ["F", "R", "I"]
    assert labels["4316"]["target_labels"] == ["1", "9", "8"]


def test_e08_outputs_use_canonical_result_stem(tmp_path: Path, monkeypatch) -> None:
    matrices = {
        "example_01": (
            np.array([[0.5]]),
            np.array([[0.5]]),
            np.array([[0.5]]),
            1,
        )
    }
    monkeypatch.setattr(
        analysis,
        "runs",
        lambda _client, _group, conditions: {
            condition: [object()] for condition in conditions
        },
    )
    monkeypatch.setattr(analysis, "_matrices", lambda *_args: matrices)
    monkeypatch.setattr(analysis, "_labels", lambda *_args: {})

    outputs = analysis.render(object(), "band", tmp_path / "ds2_e08_imp.png")

    assert [path.name for path in outputs] == [
        "ds2_e08_imp_attention_alignment_example_01.png",
        "ds2_e08_imp_attention_alignment.csv",
        "ds2_e08_imp_attention_alignment_greedy_example_01.png",
        "ds2_e08_imp_attention_alignment_greedy.csv",
    ]
    assert all(path.is_file() for path in outputs)
