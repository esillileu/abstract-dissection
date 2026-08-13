from exp.deepscratch.original_runtime.promoted_executor import _metric_name, _metric_rows


def test_promoted_perplexity_metrics_preserve_dataset_split() -> None:
    assert _metric_name("perplexity", split="valid") == "valid/perplexity"
    assert _metric_name("perplexity", split="test") == "test/perplexity"


def test_promoted_original_accuracy_uses_epoch_axis(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text(
        "update,epoch,split,accuracy,loss\n"
        "0,0,train,0.10,1.5\n"
        "3,1,train,0.20,1.2\n",
        encoding="utf-8",
    )

    rows, _ = _metric_rows(tmp_path)

    assert [(step, metric) for step, metric, _ in rows] == [
        (0, "train/accuracy"),
        (0, "train/loss"),
        (1, "train/accuracy"),
        (3, "train/loss"),
    ]
