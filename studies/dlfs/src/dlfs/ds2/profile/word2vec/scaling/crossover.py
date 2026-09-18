from __future__ import annotations

import itertools


def _crossovers(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result = {
        "CBOW": _crossover(rows, "CBOW", "NegativeSampling"),
        "CBOW-Fused": _crossover(rows, "CBOW", "FusedNegativeSampling"),
        "SkipGram": _crossover(rows, "SkipGram", "NegativeSampling"),
        "SkipGram-Fused": _crossover(
            rows,
            "SkipGram",
            "FusedNegativeSampling",
        ),
    }
    return {key: value for key, value in result.items() if value["comparisons"]}


def summarize_crossovers(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize full-softmax and sampling crossover observations."""
    return _crossovers(rows)


def _crossover(
    rows: list[dict[str, object]],
    model: str,
    sampling_objective: str,
) -> dict[str, object]:
    by_key = {
        (int(row["vocab_size"]), str(row["objective"])): row
        for row in rows
        if row["model"] == model and row["status"] == "ok"
    }
    comparisons = []
    for vocab_size in sorted({key[0] for key in by_key}):
        ns = by_key.get((vocab_size, sampling_objective))
        fs = by_key.get((vocab_size, "FullSoftmax"))
        if ns is None or fs is None:
            continue
        ns_ms = float(ns["update_ms"])
        fs_ms = float(fs["update_ms"])
        ns_lower = ns["ci95_lower_ms"]
        ns_upper = ns["ci95_upper_ms"]
        fs_lower = fs["ci95_lower_ms"]
        fs_upper = fs["ci95_upper_ms"]
        if (
            ns_upper is not None
            and fs_lower is not None
            and float(ns_upper) < float(fs_lower)
        ):
            confidence_winner = sampling_objective
        elif (
            fs_upper is not None
            and ns_lower is not None
            and float(fs_upper) < float(ns_lower)
        ):
            confidence_winner = "FullSoftmax"
        else:
            confidence_winner = "Inconclusive"
        comparisons.append(
            {
                "vocab_size": vocab_size,
                "negative_sampling_ms": ns_ms,
                "full_softmax_ms": fs_ms,
                "negative_sampling_speedup": fs_ms / ns_ms,
                "point_estimate_winner": (
                    sampling_objective if ns_ms < fs_ms else "FullSoftmax"
                ),
                "confidence_winner": confidence_winner,
            }
        )
    first_observed = next(
        (
            comparison["vocab_size"]
            for comparison in comparisons
            if comparison["point_estimate_winner"] == sampling_objective
        ),
        None,
    )
    first_confirmed = next(
        (
            comparison["vocab_size"]
            for comparison, following in itertools.pairwise(comparisons)
            if comparison["confidence_winner"] == sampling_objective
            and following["confidence_winner"] == sampling_objective
        ),
        None,
    )
    return {
        "sampling_objective": sampling_objective,
        "first_observed_negative_sampling_win_vocab_size": first_observed,
        "first_confirmed_negative_sampling_win_vocab_size": first_confirmed,
        "confirmation_rule": (
            "non-overlapping 95% mean confidence intervals favor "
            f"{sampling_objective} at this and the next measured vocabulary size"
        ),
        "comparisons": comparisons,
    }


def _render_crossovers(
    device: str,
    crossovers: object,
) -> str:
    assert isinstance(crossovers, dict)
    lines = [f"\n# {device} vocabulary-size scaling crossover"]
    for model in ("CBOW", "CBOW-Fused", "SkipGram", "SkipGram-Fused"):
        summary = crossovers.get(model)
        if summary is None:
            continue
        assert isinstance(summary, dict)
        first = summary["first_confirmed_negative_sampling_win_vocab_size"]
        objective = str(summary["sampling_objective"])
        lines.append(
            f"- {model}: "
            + (
                f"no confirmed {objective} crossover"
                if first is None
                else f"confirmed {objective} crossover at V={int(first):,}"
            )
        )
    return "\n".join(lines)
