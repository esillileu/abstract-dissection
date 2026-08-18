from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from exp.deepscratch.identity import Variant, Volume
from exp.deepscratch.execution.selection import CanonicalAttemptSelector


def _uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_selector_prefers_canonical_and_excludes_alternate(tmp_path: Path) -> None:
    client = MlflowClient(_uri(tmp_path))
    legacy = client.create_experiment("ds2_original")
    canonical = client.create_experiment("deepscratch.ds2")
    common = {
        "run.type": "seed_trial",
        "experiment.id": "e05",
        "condition.id": "BETTER-RNNLM",
        "master_seed": "4",
    }
    native = client.create_run(legacy, start_time=1, tags=common)
    client.set_terminated(native.info.run_id)
    alternate = client.create_run(legacy, start_time=3, tags={
        **common, "transfer.import.disposition": "imported-alternate"
    })
    client.set_terminated(alternate.info.run_id)
    primary = client.create_run(canonical, start_time=2, tags={
        **common,
        "implementation.variant": "original",
        "result.durable_complete": "true",
    })
    client.set_terminated(primary.info.run_id)

    selector = CanonicalAttemptSelector(client)
    selected = selector.select(
        Volume.DS2,
        Variant.ORIGINAL,
        study_id="e05",
        condition_ids=("BETTER-RNNLM",),
        seed=4,
    )
    assert selected is not None and selected.run_id == primary.info.run_id
    with pytest.raises(ValueError, match="not in the requested coordinate"):
        selector.select(
            Volume.DS2,
            Variant.ORIGINAL,
            study_id="e05",
            condition_ids=("BETTER-RNNLM",),
            seed=4,
            run_id=alternate.info.run_id,
        )


def test_selector_fetches_each_variant_inventory_only_once(tmp_path: Path) -> None:
    source = MlflowClient(_uri(tmp_path))
    source.create_experiment("deepscratch.ds2")
    source.create_experiment("ds2")

    class CountingClient:
        def __init__(self, client) -> None:
            self.client = client
            self.search_count = 0

        def search_runs(self, *args, **kwargs):
            self.search_count += 1
            return self.client.search_runs(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.client, name)

    client = CountingClient(source)
    selector = CanonicalAttemptSelector(client)

    selector.attempts(Volume.DS2, Variant.IMPLEMENTED)
    selector.attempts(Volume.DS2, Variant.IMPLEMENTED)

    assert client.search_count == 1
