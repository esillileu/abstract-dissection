"""Integration test for NNLM resume script."""

import pytest

from repro_core.context import RuntimePaths


def test_resume_nnlm_dry_run():
    paths = RuntimePaths.from_environment()
    run_id = "e39307353a574aa9a2ab20eb540d432e"
    run_root = paths.staging_root / "exp" / "f2" / "w2v1" / "tracked" / run_id

    if not run_root.is_dir():
        pytest.skip(f"Staging run {run_id} does not exist in active staging area")

    # Import and run resume in dry-run mode
    import sys

    sys.path.insert(0, str(paths.repo_root / "tools"))
    from resume_nnlm import resume_training

    config_path = (
        paths.repo_root / "studies/f2/src/f2/suites/w2v1/config/e02_table3.yaml"
    )
    resume_training(
        run_id=run_id,
        config_path=config_path,
        atomic_run_id="wmt--nnlm-d640-h640-n8-w320m",
        seed=1,
        dry_run=True,
        publish=False,
    )
