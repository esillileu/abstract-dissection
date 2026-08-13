import numpy as np
import pytest

from exp.deepscratch.ds2.profile.e02 import analyze, api, modules
from exp.deepscratch.ds2.profile.e02 import update
from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.nn.model.architecture import (
    OneHotCBOW,
    OneHotCBOWBatchAdapter,
    OneHotSkipGram,
    OneHotSkipGramBatchAdapter,
)


def test_profile_registers_embedding_and_onehot_full_softmax_conditions() -> None:
    assert "original-cbow-fs" in update.CONDITIONS
    assert "original-skipgram-fs" in update.CONDITIONS
    assert "original-cbow-onehot-fs" in update.CONDITIONS
    assert "original-skipgram-onehot-fs" in update.CONDITIONS
    assert "implemented-cbow-onehot-fs" in update.CONDITIONS
    assert "implemented-skipgram-onehot-fs" in update.CONDITIONS
    assert "implemented-cbow-fused-ns" in update.CONDITIONS
    assert "implemented-skipgram-fused-ns" in update.CONDITIONS
    assert update.CONDITIONS == (
        "original-cbow-onehot-fs",
        "original-cbow-fs",
        "original-cbow-ns",
        "original-skipgram-onehot-fs",
        "original-skipgram-fs",
        "original-skipgram-ns",
        "implemented-cbow-onehot-fs",
        "implemented-cbow-fs",
        "implemented-cbow-ns",
        "implemented-cbow-fused-ns",
        "implemented-skipgram-onehot-fs",
        "implemented-skipgram-fs",
        "implemented-skipgram-ns",
        "implemented-skipgram-fused-ns",
    )


@pytest.mark.parametrize(
    ("condition", "expected_model"),
    (
        ("implemented-cbow-fused-ns", "CBOW"),
        ("implemented-skipgram-fused-ns", "SkipGram"),
    ),
)
def test_profile_fused_condition_runs_one_cpu_update(
    condition: str,
    expected_model: str,
) -> None:
    backend = make_backend(
        BackendConfig(device="cpu", dtype="float32", seed=1)
    )
    corpus = np.arange(11, dtype=np.int32)
    contexts = np.tile(np.arange(10, dtype=np.int32), (4, 1))
    targets = np.arange(4, dtype=np.int32)
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

    assert np.isfinite(loss.data)
    assert model_name == expected_model
    assert objective_name == "FusedNegativeSampling"
    assert implementation == "implemented"


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

        loss_value = loss.data if hasattr(loss, "data") else loss
        assert np.isfinite(loss_value)
        assert model_name in {"CBOW", "SkipGram"}
        assert objective_name == "FullSoftmax"
        assert implementation == "original"


def test_profile_onehot_full_softmax_conditions_run_one_cpu_update() -> None:
    update._install_original_imports("cpu")
    backend = make_backend(
        BackendConfig(device="cpu", dtype="float32", seed=1)
    )
    corpus = np.arange(11, dtype=np.int32)
    contexts = np.tile(np.arange(10, dtype=np.int32), (4, 1))
    targets = np.arange(4, dtype=np.int32)

    conditions = (
        "original-cbow-onehot-fs",
        "original-skipgram-onehot-fs",
        "implemented-cbow-onehot-fs",
        "implemented-skipgram-onehot-fs",
    )
    for condition in conditions:
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

        loss_value = loss.data if hasattr(loss, "data") else loss
        assert np.isfinite(loss_value)
        assert model_name in {"CBOW", "SkipGram"}
        assert objective_name == "FullSoftmax"
        assert implementation in {"original", "implemented"}
        if implementation == "implemented":
            expected_model = (
                OneHotCBOW if model_name == "CBOW" else OneHotSkipGram
            )
            expected_adapter = (
                OneHotCBOWBatchAdapter
                if model_name == "CBOW"
                else OneHotSkipGramBatchAdapter
            )
            assert isinstance(workload.model, expected_model)
            assert isinstance(workload.adapter, expected_adapter)


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

    assert "Original Emb. FS" in table
    assert "Original One-hot FS" in table
    assert "Implemented One-hot FS" in table
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
    assert "implemented-cbow-onehot-fs" in {
        str(row["baseline"]) for row in comparisons
    } | {
        str(row["candidate"]) for row in comparisons
    }
    assert "implemented-skipgram-onehot-fs" in {
        str(row["baseline"]) for row in comparisons
    } | {
        str(row["candidate"]) for row in comparisons
    }


def test_full_profile_report_includes_onehot_conditions_and_comparison() -> None:
    rows = [
        {
            "condition": condition,
            "cold_ms_per_update": float(index + 1),
            "steady_event_p50_ms_per_update": float(index + 1),
            "steady_event_p95_ms_per_update": float(index + 2),
            "mean_ms_per_update": float(index + 1),
            "stdev_ms_per_update": 0.1,
            "samples_per_second": 100.0,
            "estimated_seconds_per_epoch": float(index + 2),
            "estimated_repeat_stdev_seconds_per_epoch": 0.2,
            "estimated_seconds_total": float(index + 3),
            "estimated_repeat_stdev_seconds_total": 0.3,
            "phase_ms_per_update": {},
        }
        for index, condition in enumerate(update.CONDITIONS)
    ]
    report = analyze.render_report(
        {
            "metadata": {
                "device": "cpu",
                "device_name": "cpu",
                "cupy_version": None,
            },
            "results": rows,
        }
    )

    assert "`implemented-cbow-onehot-fs`" in report
    assert "`implemented-skipgram-onehot-fs`" in report
    assert "## One-hot input projection cost" in report
    assert "Implemented CBOW FS: embedding → one-hot" in report


def test_onehot_conditions_are_available_to_module_profiler() -> None:
    update._install_original_imports("cpu")
    backend = make_backend(
        BackendConfig(device="cpu", dtype="float32", seed=1)
    )
    corpus = np.arange(11, dtype=np.int32)
    contexts = np.tile(np.arange(10, dtype=np.int32), (4, 1))
    targets = np.arange(4, dtype=np.int32)

    rows = modules.profile_modules(
        "implemented-cbow-onehot-fs",
        corpus=corpus,
        contexts=contexts,
        targets=targets,
        backend=backend,
        batch_size=2,
        components=("batch_adapter",),
        warmup_iterations=0,
        measured_iterations=1,
    )

    assert modules.CONDITIONS is update.CONDITIONS
    assert rows[0]["condition"] == "implemented-cbow-onehot-fs"
    assert rows[0]["component"] == "batch_adapter"


@pytest.mark.parametrize(
    "condition",
    (
        "implemented-cbow-fused-ns",
        "implemented-skipgram-fused-ns",
    ),
)
def test_fused_condition_is_available_to_module_profiler(
    condition: str,
) -> None:
    backend = make_backend(
        BackendConfig(device="cpu", dtype="float32", seed=1)
    )
    corpus = np.arange(11, dtype=np.int32)
    contexts = np.tile(np.arange(10, dtype=np.int32), (4, 1))
    targets = np.arange(4, dtype=np.int32)

    rows = modules.profile_modules(
        condition,
        corpus=corpus,
        contexts=contexts,
        targets=targets,
        backend=backend,
        batch_size=2,
        components=("fused_forward_loss", "fused_backward", "optimizer"),
        warmup_iterations=0,
        measured_iterations=1,
    )

    assert [row["component"] for row in rows] == [
        "fused_forward_loss",
        "fused_backward",
        "optimizer",
    ]
