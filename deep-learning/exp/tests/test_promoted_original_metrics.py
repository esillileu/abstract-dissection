from exp.original.promoted_executor import _metric_name


def test_promoted_perplexity_metrics_preserve_dataset_split() -> None:
    assert _metric_name("perplexity", split="valid") == "valid/perplexity"
    assert _metric_name("perplexity", split="test") == "test/perplexity"
