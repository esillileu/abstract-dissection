from pathlib import Path

import matplotlib.pyplot as plt

from exp.deepscratch.original_runtime.cache_protocol import save_csv


def test_original_e07_combines_all_conditions_in_one_figure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from exp.deepscratch.ds2.original.native_analysis import e07

    root = tmp_path / "original"
    for index, trial_id in enumerate(e07.TRIAL_IDS):
        save_csv(
            root / "data" / "e07" / trial_id / "metrics.csv",
            [
                {"epoch": 0, "accuracy": index / 10},
                {"epoch": 1, "accuracy": (index + 1) / 10},
            ],
        )

    saved_paths = []
    monkeypatch.setattr(e07, "save", saved_paths.append)

    outputs = e07.render(root, root / "image")
    axis = plt.gca()

    assert outputs == [root / "image" / "e07_date_seq2seq.png"]
    assert saved_paths == outputs
    assert len(axis.lines) == 3
    assert [line.get_label() for line in axis.lines] == list(e07.LABELS)
    assert all(line.get_marker() == "o" for line in axis.lines)
    assert axis.get_xlabel() == "epochs"
    assert axis.get_ylabel() == "accuracy"
    assert axis.get_ylim() == (-0.05, 1.05)
    assert axis.get_title() == ""
    assert axis.get_legend() is not None

    plt.close()
