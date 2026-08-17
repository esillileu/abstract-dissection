from mlprosection_mlflow import RunIdentity, build_tags, canonical_json, flatten_dict, make_condition_key, make_run_key
from mlprosection_mlflow import build_schema_metrics


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


def test_build_tags_includes_a_human_readable_model_name() -> None:
    identity = RunIdentity(1, "mlprosection", ("e02",), "MLP-ADAM-HE", "g02", "RC-MLP", "mlp-v1", "condition", "run", 7)

    tags = build_tags(
        identity,
        {"kind": "test", "dataset": {"id": "MNIST"}, "model": {"name": "MLP4Hidden", "family": "mlp", "task_type": "classification"}},
        {"commit": "abc", "branch": "main", "dirty": False, "repository": "repo", "entrypoint": "yaml", "diff_sha256": "diff"},
        None,
    )

    assert tags["model.name"] == "MLP4Hidden"


def test_build_tags_uses_declared_device_when_result_has_no_model() -> None:
    identity = RunIdentity(
        1, "mlprosection", ("e11",), "PF-VSCALE-CBOW-NS", "PF02",
        "profile", "profile-v1", "condition", "run", 0,
    )
    tags = build_tags(
        identity,
        {
            "kind": "performance_profile",
            "numerics": {"backend": "cupy", "device": "cuda:1"},
        },
        {
            "commit": "abc", "branch": "main", "dirty": False,
            "repository": "repo", "entrypoint": "yaml",
        },
        None,
    )
    assert tags["runtime.backend"] == "cupy"
    assert tags["runtime.device_type"] == "cuda"


def test_parent_group_key_is_stable_across_config_representation_changes() -> None:
    identity = RunIdentity(
        1, "mlprosection", ("e07",), "CNN-DEEP-BOOK", "GT07",
        "ds1-gt07-deep-cnn", "mnist-deepcnn-6conv-fc50", "condition", "run", 4,
    )
    git_info = {
        "commit": "abc", "branch": "main", "dirty": False, "repository": "repo",
        "entrypoint": "yaml", "diff_sha256": "diff",
    }
    old_tags = build_tags(
        identity,
        {
            "kind": "test",
            "model": {"alias": "DeepCNN"},
            "loss": {"name": "SoftmaxWithLoss", "reduction": "mean"},
            "checkpoint": {"format": "npz"},
        },
        git_info,
        None,
    )
    new_tags = build_tags(
        identity,
        {
            "kind": "test",
            "model": {"name": "DeepCNN"},
            "objective": {"name": "SoftmaxCrossEntropy", "reduction": "mean"},
            "checkpoint": {"format": "v2"},
        },
        git_info,
        None,
    )

    assert old_tags["condition.group.key"] == new_tags["condition.group.key"]


def test_parent_group_key_distinguishes_reused_atomic_run_names() -> None:
    git_info = {
        "commit": "abc", "branch": "main", "dirty": False, "repository": "repo",
        "entrypoint": "yaml", "diff_sha256": "diff",
    }
    optimizer_identity = RunIdentity(
        1, "mlprosection", ("e02",), "MLP-SGD-HE", "GT02",
        "optimizer-comparison", "mlp-v1", "condition-a", "run-a", 1,
    )
    initializer_identity = RunIdentity(
        1, "mlprosection", ("e04",), "MLP-SGD-HE", "GT04",
        "initializer-comparison", "mlp-v1", "condition-b", "run-b", 1,
    )

    optimizer_tags = build_tags(optimizer_identity, {"kind": "test"}, git_info, None)
    initializer_tags = build_tags(initializer_identity, {"kind": "test"}, git_info, None)

    assert optimizer_tags["condition.group.key"] != initializer_tags["condition.group.key"]


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
