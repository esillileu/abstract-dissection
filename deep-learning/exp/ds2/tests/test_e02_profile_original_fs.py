import numpy as np

from exp.ds2.profile.e02 import analyze, api
from exp.ds2.profile.e02 import update
from mlprosection.core.backend import BackendConfig, make_backend


def test_profile_registers_original_full_softmax_conditions() -> None:
    assert "original-cbow-fs" in update.CONDITIONS
    assert "original-skipgram-fs" in update.CONDITIONS


def test_profile_original_full_softmax_conditions_run_one_cpu_update() -> None:
    update._install_original_imports("cpu")
    backend = make_backend(
        BackendConfig(device="cpu", dtype="float32", seed=1)
    )
    corpus = np.arange(11, dtype=np.int32)
    contexts = np.tile(np.arange(10, dtype=np.int32), (4, 1))
    targets = np.arange(4, dtype=np.int32)

    for condition in ("original-cbow-fs", "original-skipgram-fs"):
        workload, model_name, objective_name, implementation = (
            update._build_condition(
                condition,
                corpus=corpus,
                contexts=contexts,
                targets=targets,
                backend=backend,
            )
        )
        loss = workload.update(workload.contexts, workload.targets)

        assert np.isfinite(loss)
        assert model_name in {"CBOW", "SkipGram"}
        assert objective_name == "FullSoftmax"
        assert implementation == "original"


def test_profile_summary_and_comparisons_include_original_full_softmax() -> None:
    update_rows = [
        {
            "condition": condition,
            "mean_ms_per_update": float(index + 1),
            "stdev_ms_per_update": 0.1,
            "estimated_seconds_per_epoch": float(index + 2),
            "estimated_repeat_stdev_seconds_per_epoch": 0.2,
            "estimated_seconds_total": float(index + 3),
            "estimated_repeat_stdev_seconds_total": 0.3,
        }
        for index, condition in enumerate(update.CONDITIONS)
    ]
    table = api.render_summary_table(
        device="cpu",
        model="CBOW",
        update_rows=update_rows,
        module_rows=[],
    )
    comparisons = analyze.build_comparisons({"results": update_rows})

    assert "Original FS" in table
    assert "original-cbow-fs" in {
        str(row["baseline"]) for row in comparisons
    } | {
        str(row["candidate"]) for row in comparisons
    }
    assert "original-skipgram-fs" in {
        str(row["baseline"]) for row in comparisons
    } | {
        str(row["candidate"]) for row in comparisons
    }
