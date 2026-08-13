from pathlib import Path

import matplotlib.pyplot as plt

from exp.deepscratch.original_runtime.cache_protocol import save_csv


def test_original_e06_combines_all_conditions_in_one_figure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from exp.deepscratch.ds2.original.native_analysis import e06

    root = tmp_path / "original"
    for index, trial_id in enumerate(e06.TRIAL_IDS):
        save_csv(
            root / "data" / "e06" / trial_id / "metrics.csv",
            [
                {"epoch": 0, "accuracy": index / 10},
                {"epoch": 1, "accuracy": (index + 1) / 10},
            ],
        )

    saved_paths = []
    monkeypatch.setattr(e06, "save", saved_paths.append)

    outputs = e06.render(root, root / "image")
    axis = plt.gca()

    assert outputs == [root / "image" / "e06_addition_seq2seq.png"]
    assert saved_paths == outputs
    assert len(axis.lines) == 4
    assert [line.get_label() for line in axis.lines] == list(e06.LABELS)
    assert axis.get_xlabel() == "epochs"
    assert axis.get_ylabel() == "accuracy"
    assert axis.get_ylim() == (0.0, 1.0)
    assert axis.get_title() == ""
    assert axis.get_legend() is not None

    plt.close()
