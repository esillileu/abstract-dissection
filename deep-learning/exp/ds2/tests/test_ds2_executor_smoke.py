from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exp.ds2.executor import (
    AttentionAlignmentObservationExecutor,
    LanguageModelExecutor,
    Seq2SeqExecutor,
    Word2VecExecutor,
)
from exp.ds2.spec import parse_run_spec
from mlprosection.experiment import ExperimentContext
from mlprosection.nn.model.architecture import AttentionSeq2seq


class _ProgressRecorder:
    def __init__(self) -> None:
        self.total: int | None = None
        self.completed = 0
        self.updates: list[int] = []

    def set_total_updates(self, total: int, *, completed: int = 0) -> None:
        self.total = total
        self.completed = completed

    def advance_to(self, update: int, *, epoch: int = 0) -> None:
        self.updates.append(update)

    def on_update(self, event) -> None:
        self.advance_to(event.update, epoch=event.epoch)

    def on_epoch(self, _event) -> None:
        return None

    def on_train_end(self, _event) -> None:
        return None


def _context(root: Path, progress=None) -> ExperimentContext:
    return ExperimentContext(
        metadata={
            "artifact_root": root,
            "checkpoint_root": root / "checkpoints",
            **({} if progress is None else {"progress_reporter": progress}),
        }
    )


def _cpu_overrides() -> dict[str, object]:
    return {
        "numerics": {
            "backend": "numpy",
            "device": "cpu",
            "dtype": "float32",
            "deterministic": True,
        },
        "tracking": {"enabled": False},
        "profiling": {"enabled": False},
    }


def _sequence_data() -> dict[str, object]:
    train_x = np.array(
        [[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 3, 2]],
        dtype=np.int64,
    )
    train_t = np.array(
        [[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 1, 3, 2]],
        dtype=np.int64,
    )
    return {
        "train": (train_x, train_t),
        "test": (train_x[:2], train_t[:2]),
        "char_to_id": {"_": 0, "a": 1, "b": 2, "c": 3},
        "id_to_char": {0: "_", 1: "a", 2: "b", 3: "c"},
    }


def test_ds2_word2vec_config_runs_one_epoch(tmp_path: Path) -> None:
    spec = parse_run_spec(
        "exp/ds2/config/e01_toy_word2vec.yaml",
        atomic_run_id="W2V-TOY-CBOW-FULL",
        overrides={
            **_cpu_overrides(),
            "budget": {"max_epochs": 1},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1

    result = Word2VecExecutor().run(config, _context(tmp_path))

    assert result.metrics["final/status/success"] == 1.0
    assert result.metrics["final/system/total_updates"] > 0
    assert "final/train/book_loss" in result.metrics
    assert any(
        metric == "update/train/book_loss"
        for _step, metric, _value in result.metric_rows
    )
    assert (tmp_path / "updates.csv").is_file()


def test_ds2_skipgram_full_softmax_runs_grouped_contexts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    corpus = np.array([0, 1, 2, 3] * 6, dtype=np.int64)
    monkeypatch.setattr(
        "exp.ds2.executor.load_ptb",
        lambda: {
            "train": corpus,
            "valid": corpus[:8],
            "test": corpus[:8],
            "word_to_id": {str(index): index for index in range(4)},
            "id_to_word": {index: str(index) for index in range(4)},
        },
    )
    spec = parse_run_spec(
        "exp/ds2/config/e02_ptb_word2vec.yaml",
        atomic_run_id="W2V-PTB-SKIPGRAM-FULL",
        overrides={
            **_cpu_overrides(),
            "loader": {"batch_size": 2},
            "model": {"embedding_size": 3},
            "budget": {"max_epochs": 1, "max_updates": 1},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1

    result = Word2VecExecutor().run(config, _context(tmp_path))

    assert result.metrics["final/status/success"] == 1.0
    assert result.metrics["final/system/total_updates"] == 1.0
    assert "final/train/book_loss" in result.metrics


def test_ds2_language_model_config_runs_one_epoch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    corpus = np.array([0, 1, 2, 3] * 4, dtype=np.int64)
    monkeypatch.setattr(
        "exp.ds2.executor.load_ptb",
        lambda: {
            "train": corpus,
            "valid": corpus[:8],
            "test": corpus[:8],
            "word_to_id": {str(index): index for index in range(4)},
            "id_to_word": {index: str(index) for index in range(4)},
        },
    )
    spec = parse_run_spec(
        "exp/ds2/config/e03_small_rnnlm.yaml",
        atomic_run_id="LM-SMALL-RNN",
        overrides={
            **_cpu_overrides(),
            "dataset": {"train_limit": 12},
            "loader": {"batch_size": 2, "time_size": 2},
            "model": {"wordvec_size": 3, "hidden_size": 4},
            "budget": {"max_epochs": 1},
            "recording": {"source_curve": None, "evaluations": []},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1
    config["checkpoint"]["save_best"] = True
    config["evaluation"].update({
        "valid_every_epochs": 1,
        "test_at_end": True,
    })
    progress = _ProgressRecorder()

    result = LanguageModelExecutor().run(config, _context(tmp_path, progress))

    assert result.metrics["final/status/success"] == 1.0
    assert result.metrics["final/system/total_updates"] > 0
    assert progress.total == 2
    assert progress.updates == [1, 2]
    assert (tmp_path / "updates.csv").is_file()
    assert (tmp_path / "checkpoints" / "best.json").is_file()
    assert "final/test/perplexity" in result.metrics


def test_ds2_seq2seq_config_runs_one_epoch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "exp.ds2.executor.load_sequence",
        lambda *_args, **_kwargs: _sequence_data(),
    )
    spec = parse_run_spec(
        "exp/ds2/config/e06_addition_seq2seq.yaml",
        atomic_run_id="SEQA-VAN-FWD",
        overrides={
            **_cpu_overrides(),
            "loader": {"batch_size": 2},
            "model": {"wordvec_size": 3, "hidden_size": 4},
            "budget": {"max_epochs": 1},
            "recording": {
                "predictions": {"split": "test", "count": 1, "decode": "greedy"}
            },
            "checkpoint": {"save_best": False},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1
    progress = _ProgressRecorder()

    result = Seq2SeqExecutor().run(config, _context(tmp_path, progress))

    assert result.metrics["final/status/success"] == 1.0
    assert result.metrics["final/system/total_updates"] == 2.0
    assert progress.total == 2
    assert progress.updates == [1, 2]
    assert (tmp_path / "evaluations.csv").is_file()
    assert (tmp_path / "observations" / "predictions.csv").is_file()


def test_ds2_attention_observation_loads_checkpoint_and_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "exp.ds2.executor.load_sequence",
        lambda *_args, **_kwargs: _sequence_data(),
    )
    checkpoint = tmp_path / "source-checkpoint"
    checkpoint.mkdir()
    model = AttentionSeq2seq(
        vocab_size=4,
        wordvec_size=3,
        hidden_size=4,
        backend="cpu",
    )
    model.save_params_npz(checkpoint / "model_parameters.npz")
    (checkpoint / "manifest.json").write_text(
        json.dumps({"schema_version": 2}),
        encoding="utf-8",
    )
    spec = parse_run_spec(
        "exp/ds2/config/e08_attention_alignment.yaml",
        atomic_run_id="ATTENTION-ALIGNMENT",
        overrides={
            **_cpu_overrides(),
            "model": {"wordvec_size": 3, "hidden_size": 4},
            "recording": {
                "predictions": {"split": "test", "count": 1, "decode": "greedy"},
                "attention": {
                    "selection_seed": 1984,
                    "count": 1,
                    "decode": "greedy",
                },
            },
            "checkpoint": {"source_path": str(checkpoint)},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1

    result = AttentionAlignmentObservationExecutor().run(
        config,
        _context(tmp_path / "observation"),
    )

    assert result.metrics["final/status/success"] == 1.0
    observations = tmp_path / "observation" / "observations"
    assert (observations / "attention.csv").is_file()
    assert (observations / "attention_render.json").is_file()
