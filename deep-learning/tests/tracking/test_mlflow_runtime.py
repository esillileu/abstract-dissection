from types import SimpleNamespace

from mlprosection_mlflow.runtime import build_profiling_history_rows, get_or_create_condition_parent, metric_batches


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


def test_profiling_history_projects_epoch_runtime_and_throughput() -> None:
    rows = build_profiling_history_rows({
        "runtime.epoch.0.train_duration_ms": 1_500,
        "runtime.epoch.0.eval_duration_ms": 200,
        "throughput.epoch.0.train_samples_per_s": 512,
        "memory.epoch.0.train.start.cpu.rss_bytes": 64,
    })

    assert rows == [
        ("epoch", 1, "runtime/train_duration_s", 1.5),
        ("epoch", 1, "runtime/eval_duration_s", 0.2),
        ("epoch", 1, "throughput/train_samples_per_s", 512.0),
        ("epoch", 1, "memory/train_start/cpu_rss_bytes", 64.0),
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
