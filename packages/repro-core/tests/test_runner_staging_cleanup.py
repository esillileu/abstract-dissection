from pathlib import Path

from repro_core.execution.runner import _remove_durable_staging


def test_remove_durable_staging_preserves_checkpoints(tmp_path: Path) -> None:
    staging = tmp_path / "staging_run"
    staging.mkdir()
    (staging / "metrics").mkdir()
    (staging / "metrics" / "obs.csv").write_text("token,loss\n")
    (staging / "scratch.tmp").write_text("data")

    ckpt = staging / "checkpoints"
    ckpt.mkdir()
    (ckpt / "latest.json").write_text("{}")
    (ckpt / "weights.npy").write_text("weights")

    _remove_durable_staging(staging)

    assert staging.is_dir()
    assert ckpt.is_dir()
    assert (ckpt / "latest.json").is_file()
    assert (ckpt / "weights.npy").is_file()
    assert not (staging / "metrics").exists()
    assert not (staging / "scratch.tmp").exists()


def test_remove_durable_staging_removes_completely_if_no_checkpoints(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging_run"
    staging.mkdir()
    (staging / "metrics").mkdir()
    (staging / "metrics" / "obs.csv").write_text("token,loss\n")

    _remove_durable_staging(staging)

    assert not staging.exists()
