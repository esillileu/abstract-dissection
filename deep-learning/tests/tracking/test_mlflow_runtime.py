from mlprosection_mlflow.runtime import metric_batches


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
