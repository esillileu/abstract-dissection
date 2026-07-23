import logging
from types import SimpleNamespace

from mlprosection_mlflow.runtime import RuntimeOptions, _Sink, _silence_mlflow_progress_logs, build_profiling_metric_rows, get_or_create_condition_parent, metric_batches


def test_completed_checkpoint_upload_is_enabled_but_eval_upload_is_disabled_by_default() -> None:
    options = RuntimeOptions(tracking_uri="http://example.invalid", experiment_name="example")

    assert options.upload_checkpoint is True
    assert options.upload_eval_checkpoints is False


def test_mlflow_lifecycle_info_is_silenced_without_hiding_warnings() -> None:
    logger = logging.getLogger("mlflow.tracking.fluent")
    original_level = logger.level
    try:
        _silence_mlflow_progress_logs()
        assert logger.isEnabledFor(logging.INFO) is False
        assert logger.isEnabledFor(logging.WARNING) is True
    finally:
        logger.setLevel(original_level)


class _ArtifactClient:
    def __init__(self) -> None:
        self.files = []
        self.trees = []

    def log_artifact(self, run_id, path, artifact_path) -> None:
        self.files.append((run_id, path, artifact_path))

    def log_artifacts(self, run_id, path, artifact_path) -> None:
        self.trees.append((run_id, path, artifact_path))


def test_final_checkpoint_uploads_by_default_but_eval_checkpoint_requires_opt_in(tmp_path, monkeypatch) -> None:
    final_path = tmp_path / "latest-epoch-0001"
    final_path.mkdir()
    (final_path / "model_parameters.npz").write_bytes(b"final")
    eval_path = tmp_path / "epoch-0001"
    eval_path.mkdir()
    (eval_path / "model_parameters.npz").write_bytes(b"eval")
    client = _ArtifactClient()
    sink = _Sink(RuntimeOptions(tracking_uri="http://example.invalid", experiment_name="example"), "run", {}, {})
    sink.run_id = "run-id"
    monkeypatch.setattr(sink, "_start_mlflow", lambda: client)
    sink.put(("checkpoint", (final_path, "final")))
    sink.put(("checkpoint", (eval_path, "eval")))
    sink.put(("stop", "FINISHED"))

    sink._consume()

    assert client.files == []
    assert client.trees == [
        ("run-id", str(final_path), f"checkpoints/{final_path.name}")
    ]

    enabled_client = _ArtifactClient()
    enabled_sink = _Sink(
        RuntimeOptions(
            tracking_uri="http://example.invalid",
            experiment_name="example",
            upload_eval_checkpoints=True,
        ),
        "run",
        {},
        {},
    )
    enabled_sink.run_id = "run-id"
    monkeypatch.setattr(enabled_sink, "_start_mlflow", lambda: enabled_client)
    enabled_sink.put(("checkpoint", (eval_path, "eval")))
    enabled_sink.put(("stop", "FINISHED"))

    enabled_sink._consume()

    assert enabled_client.trees == [("run-id", str(eval_path), "checkpoints/epoch-0001")]


def test_artifact_upload_uses_no_artifact_path_for_root_files(tmp_path, monkeypatch) -> None:
    root_file = tmp_path / "updates.csv"
    root_file.write_text("update,loss\n", encoding="utf-8")
    nested_file = tmp_path / "metrics" / "final.json"
    nested_file.parent.mkdir()
    nested_file.write_text("{}", encoding="utf-8")
    client = _ArtifactClient()
    sink = _Sink(RuntimeOptions(tracking_uri="http://example.invalid", experiment_name="example"), "run", {}, {})
    sink.run_id = "run-id"
    monkeypatch.setattr(sink, "_start_mlflow", lambda: client)
    sink.put(("artifact", tmp_path))
    sink.put(("stop", "FINISHED"))

    sink._consume()

    assert ("run-id", str(root_file), None) in client.files
    assert ("run-id", str(nested_file), "metrics") in client.files
    assert sink.errors == []


def test_metric_batches_bounds_each_mlflow_request() -> None:
    rows = [(index, "train/loss", float(index)) for index in range(2_001)]

    batches = metric_batches(rows, batch_size=1_000)

    assert [len(batch) for batch in batches] == [1_000, 1_000, 1]
    assert batches[0][0] == rows[0]
    assert batches[-1][-1] == rows[-1]


def test_metric_batches_rejects_an_invalid_batch_size() -> None:
    try:
        metric_batches([], batch_size=0)
    except ValueError as exc:
        assert str(exc) == "metric_batch_size must be at least 1"
    else:
        raise AssertionError("expected metric batch size validation")


def test_profiling_metrics_project_epoch_runtime_and_throughput() -> None:
    rows = build_profiling_metric_rows({
        "runtime.epoch.0.train_duration_ms": 1_500,
        "runtime.epoch.0.eval_duration_ms": 200,
        "throughput.epoch.0.train_samples_per_s": 512,
        "memory.epoch.0.train.start.cpu.rss_bytes": 64,
    })

    assert rows == [
        (1, "epoch/runtime/train_duration_s", 1.5),
        (1, "epoch/runtime/eval_duration_s", 0.2),
        (1, "epoch/throughput/train_samples_per_s", 512.0),
        (1, "epoch/memory/train_start/cpu_rss_bytes", 64.0),
    ]


class FakeClient:
    def __init__(self) -> None:
        self.parents = []
        self.created_tags = None
        self.terminated = []

    def search_runs(self, **kwargs):
        self.search_args = kwargs
        return self.parents

    def create_run(self, *, experiment_id, tags):
        self.created_experiment_id = experiment_id
        self.created_tags = tags
        parent = SimpleNamespace(info=SimpleNamespace(run_id="parent-1"))
        self.parents.append(parent)
        return parent

    def set_terminated(self, run_id, *, status):
        self.terminated.append((run_id, status))


def test_condition_parent_is_created_once_per_condition_key() -> None:
    client = FakeClient()
    child_tags = {
        "run.type": "seed_trial", "condition.key": "condition-abc",
        "atomic_run.id": "MLP-ADAM-HE", "run.key": "seed-1", "master_seed": "1", "trial.status": "running",
    }

    first = get_or_create_condition_parent(client, experiment_id="1", child_tags=child_tags)
    second = get_or_create_condition_parent(client, experiment_id="1", child_tags=child_tags)

    assert first == second == "parent-1"
    assert client.created_tags == {
        "run.type": "condition_parent", "condition.key": "condition-abc", "atomic_run.id": "MLP-ADAM-HE",
        "condition.status": "running", "condition.group.key": "condition-abc", "mlflow.runName": "MLP-ADAM-HE",
    }
    assert client.terminated == [("parent-1", "FINISHED")]


class LegacyParentClient:
    def __init__(self) -> None:
        self.parent = SimpleNamespace(
            info=SimpleNamespace(run_id="legacy-parent"),
            data=SimpleNamespace(tags={
                "run.type": "condition_parent",
                "experiment.ids": "e07",
                "execution_group.id": "GT07",
                "recipe.id": "ds1-gt07-deep-cnn",
                "structure.signature": "mnist-deepcnn-6conv-fc50",
                "atomic_run.id": "CNN-DEEP-BOOK",
                "condition.group.key": "legacy-key",
            }),
        )
        self.updated_tags = []
        self.created = False

    def search_runs(self, *, filter_string, **kwargs):
        if "condition.group.key" in filter_string:
            return []
        return [self.parent]

    def set_tag(self, run_id, key, value):
        self.updated_tags.append((run_id, key, value))

    def create_run(self, **kwargs):
        self.created = True
        raise AssertionError("a new parent must not be created")


def test_legacy_parent_is_reused_and_promoted_to_canonical_group_key() -> None:
    client = LegacyParentClient()
    child_tags = {
        "run.type": "seed_trial",
        "condition.key": "new-condition-key",
        "condition.group.key": "canonical-key",
        "experiment.ids": "e07",
        "execution_group.id": "GT07",
        "recipe.id": "ds1-gt07-deep-cnn",
        "structure.signature": "mnist-deepcnn-6conv-fc50",
        "atomic_run.id": "CNN-DEEP-BOOK",
    }

    parent_id = get_or_create_condition_parent(
        client,
        experiment_id="1",
        child_tags=child_tags,
    )

    assert parent_id == "legacy-parent"
    assert client.updated_tags == [
        ("legacy-parent", "condition.group.key", "canonical-key")
    ]
    assert client.created is False
