from mlprosection.tracking import make_condition_key, make_run_key
from mlprosection.tracking.mlflow_logger import canonical_json, flatten_dict
from mlprosection.tracking.schema import build_schema_metrics


def test_canonical_json_is_key_order_invariant() -> None:
    left = {"b": 2, "a": {"d": 4, "c": 3}}
    right = {"a": {"c": 3, "d": 4}, "b": 2}

    assert canonical_json(left) == canonical_json(right)


def test_condition_key_ignores_seed_when_seed_is_not_in_condition() -> None:
    condition = {"schema_version": 1, "code": {"git_commit": "abc"}}
    seed_a = {"master": 1}
    seed_b = {"master": 2}

    assert make_condition_key(condition) == make_condition_key(condition)
    assert make_run_key(condition, seed_a) != make_run_key(condition, seed_b)


def test_flatten_dict_uses_slash_keys() -> None:
    assert flatten_dict({"model": {"name": "SimpleCNN"}}) == {
        "model/name": "SimpleCNN"
    }


def test_build_schema_metrics_maps_runtime_memory_and_profile_metrics() -> None:
    metrics = build_schema_metrics(
        train_loss=1.0,
        test_loss=2.0,
        train_accuracy=0.5,
        test_accuracy=0.25,
        profiling_metrics={
            "runtime.train_total.mean_ms": 1000,
            "memory.run.start.cpu.rss_bytes": 10,
            "memory.run.end.cpu.rss_bytes": 20,
            "memory.peak_sampled.cpu.rss_bytes": 30,
            "runtime.profile.forward.count": 2,
            "runtime.profile.forward.mean_ms": 50,
            "runtime.profile.forward.p95_ms": 60,
        },
        total_updates=3,
        completed_epochs=1,
        samples_seen=12,
    )

    assert metrics["final/train/loss"] == 1.0
    assert metrics["final/test/accuracy"] == 0.25
    assert metrics["runtime/train_total_s"] == 1.0
    assert metrics["memory/cpu_rss_peak_sampled_bytes"] == 30
    assert metrics["profile/forward/count"] == 2
    assert metrics["profile/forward/mean_s"] == 0.05
    assert metrics["profile/forward/total_s"] == 0.1
    assert metrics["profile/forward/fraction_of_train_time"] == 0.1
    assert metrics["profile/gradient_clip/count"] == 0
