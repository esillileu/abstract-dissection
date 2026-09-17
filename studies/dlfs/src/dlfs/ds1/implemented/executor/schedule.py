from __future__ import annotations

from repro_core.context.event_executor import EvaluationRequest


def _evaluation_requests(
    *, evaluation: dict[str, object], x_train, t_train, x_valid, t_valid, x_test, t_test
) -> dict[str, EvaluationRequest]:
    raw_sources = evaluation.get("sources", ())
    if not isinstance(raw_sources, list | tuple):
        raise ValueError("evaluation.sources must be a list")
    requests: dict[str, EvaluationRequest] = {}
    for source in raw_sources:
        if not isinstance(source, dict):
            raise ValueError("evaluation source must be a mapping")
        source_id = str(source["id"])
        split = str(source["split"])
        kind = str(source["kind"])
        if split == "train":
            x, t = x_train, t_train
        elif split == "valid":
            if x_valid is None or t_valid is None:
                continue
            x, t = x_valid, t_valid
        elif split == "test":
            x, t = x_test, t_test
        else:
            raise ValueError(f"unsupported evaluation source split: {split}")
        if kind == "first_n":
            count = int(source["count"])
            x, t = x[:count], t[:count]
        elif kind != "full":
            raise ValueError(f"unsupported evaluation source kind: {kind}")
        requests[source_id] = EvaluationRequest(
            source_id, split, (x, t), ("loss", "accuracy")
        )
    return requests
