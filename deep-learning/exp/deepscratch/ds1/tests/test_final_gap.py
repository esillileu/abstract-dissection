from __future__ import annotations

from types import SimpleNamespace

import pytest

from exp.deepscratch.ds1.implemented import final_gap
from exp.deepscratch.ds1.implemented.backfill_full_train_gap import backfill_runs, latest_target_runs


class _Trainer:
    def __init__(self):
        self.results = iter(
            (
                SimpleNamespace(accuracy=0.97),
                SimpleNamespace(accuracy=0.94),
            )
        )

    def evaluate(self, *_args):
        return next(self.results)


def _run(run_id: str, *, seed: int, start_time: int, metrics=None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start_time),
        data=SimpleNamespace(
            tags={
                "atomic_run.id": "CNN-SIMPLE-BOOK",
                "execution_group.id": "GT06",
            },
            params={"seed/master": str(seed)},
            metrics=dict(metrics or {}),
        ),
    )


def test_checkpoint_gap_loads_checkpoint_and_subtracts_full_test(
    monkeypatch: pytest.MonkeyPatch,
):
    loaded = []
    monkeypatch.setattr(
        final_gap,
        "load_model_checkpoint",
        lambda path, model: loaded.append((path, model)),
    )
    model = object()

    _train, _test, metrics = final_gap.evaluate_checkpoint_gap(
        trainer=_Trainer(),
        model=model,
        checkpoint="latest.json",
        x_train=object(),
        t_train=object(),
        x_test=object(),
        t_test=object(),
    )

    assert loaded == [("latest.json", model)]
    assert metrics[final_gap.TRAIN_FULL_ACCURACY] == pytest.approx(0.97)
    assert metrics[final_gap.TRAIN_TEST_ACCURACY_GAP] == pytest.approx(0.03)


def test_latest_target_runs_keeps_newest_attempt_per_seed():
    class Client:
        def get_experiment_by_name(self, _name):
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, **_kwargs):
            return [
                _run("new", seed=1, start_time=20),
                _run("old", seed=1, start_time=10),
                _run("seed-2", seed=2, start_time=15),
            ]

    runs = latest_target_runs(
        Client(),
        targets={("GT06", "CNN-SIMPLE-BOOK")},
        seeds=set(),
    )

    assert [run.info.run_id for run in runs] == ["new", "seed-2"]


def test_backfill_is_dry_run_by_default_and_apply_logs_both_metrics(capsys):
    class Client:
        def __init__(self):
            self.logged = []
            self.tags = []

        def log_metric(self, run_id, key, value, *, step):
            self.logged.append((run_id, key, value, step))

        def set_tag(self, run_id, key, value):
            self.tags.append((run_id, key, value))

    client = Client()
    run = _run(
        "run-1",
        seed=1,
        start_time=1,
        metrics={"final/system/total_updates": 12000},
    )
    evaluations = []

    def evaluate(_run):
        evaluations.append(_run.info.run_id)
        return {
            final_gap.TRAIN_FULL_ACCURACY: 0.99,
            final_gap.TRAIN_TEST_ACCURACY_GAP: 0.01,
        }

    counts = backfill_runs(
        client,
        [run],
        apply=False,
        force=False,
        evaluator=evaluate,
    )
    assert counts == {"planned": 1, "updated": 0, "skipped": 0, "failed": 0}
    assert evaluations == []
    assert client.logged == []

    counts = backfill_runs(
        client,
        [run],
        apply=True,
        force=False,
        evaluator=evaluate,
    )
    assert counts == {"planned": 1, "updated": 1, "skipped": 0, "failed": 0}
    assert evaluations == ["run-1"]
    assert [item[1] for item in client.logged] == [
        final_gap.TRAIN_FULL_ACCURACY,
        final_gap.TRAIN_TEST_ACCURACY_GAP,
    ]
    assert all(item[3] == 12000 for item in client.logged)
    assert "would update" in capsys.readouterr().out
