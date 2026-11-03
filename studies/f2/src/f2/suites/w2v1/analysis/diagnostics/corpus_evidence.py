"""Count-based discriminability and matched-control sensitivity from one scan."""

from collections import defaultdict

import numpy as np

from .corpus_statistics import WINDOWS
from .relations import write_csv


def discriminability_evidence(corpus, config, cache, output, relations, words, proxy):
    contexts = np.load(cache / "contexts.npy", mmap_mode="r")
    index = {r: i for i, r in enumerate(config["targets"])}
    controls = {int(k): v for k, v in config["controls"].items()}
    proxy_map = {(r["category"], r["window"], int(r["row"])): r for r in proxy}
    groups = {k: list(v) for k, v in config["groups"].items()}
    groups.update(
        {
            k + "-matched-controls": [controls[r] for r in v]
            for k, v in config["groups"].items()
        }
    )
    records = []
    for w, window in enumerate(WINDOWS):
        for category, ids in groups.items():
            values = np.asarray(
                contexts[w, :, [index[r] for r in ids], :].sum(axis=1), dtype=float
            )
            # Advanced indexing puts selected targets first: target, side, context.
            mass = values.sum(axis=1)
            p = np.divide(
                values,
                mass[:, None],
                out=np.zeros_like(values),
                where=mass[:, None] > 0,
            )
            mixture = p.mean(axis=0)
            available = mixture > 0
            ratio = np.divide(
                p, mixture[None, :], out=np.zeros_like(p), where=available[None, :]
            )
            logratio = np.zeros_like(p)
            np.log2(ratio, out=logratio, where=ratio > 0)
            for i, row in enumerate(ids):
                nb = proxy_map.get((category, window, row))
                chance = float(nb["chance_accuracy"]) if nb else 0
                accuracy = float(nb["heldout_accuracy"]) if nb else 0
                records.append(
                    dict(
                        corpus=corpus,
                        category=category,
                        window=window,
                        row=row,
                        single_context_center_information_bits=float(
                            np.sum(p[i] * logratio[i])
                        ),
                        single_context_correct_center_probability=float(
                            np.sum(p[i] * ratio[i]) / len(ids)
                        ),
                        conditional_proxy_discriminability=(accuracy - chance)
                        / (1 - chance)
                        if nb
                        else 0,
                        proxy_available=nb is not None,
                        candidate_centers=len(ids),
                    )
                )
    lookup = {(r["category"], r["window"], r["row"]): r for r in records}
    token_rows = {r["word"]: int(r["row"]) for r in words}
    masses = {(int(r["row"]), r["window"]): float(r["context_events"]) for r in words}
    for r in relations:
        a, b = token_rows[r["word_a"]], token_rows[r["word_b"]]
        x = lookup[r["category"], r["window"], a]
        y = lookup[r["category"], r["window"], b]
        for feature in (
            "single_context_center_information_bits",
            "single_context_correct_center_probability",
            "conditional_proxy_discriminability",
        ):
            r[feature] = (x[feature] + y[feature]) / 2
        r["context_mass_a"] = masses[a, r["window"]]
        r["context_mass_b"] = masses[b, r["window"]]
    write_csv(output / "context_discriminability.csv", records)
    write_csv(output / "relation_features.csv", relations)
    return records


def mass_matched_negatives(corpus, relations, output):
    rng = np.random.default_rng(20261011)
    records = []
    summaries = []
    for window in WINDOWS:
        for category in sorted({r["category"] for r in relations}):
            positive = [
                r
                for r in relations
                if r["category"] == category
                and r["window"] == window
                and r["kind"] == "relation"
            ]
            negatives = [
                r
                for r in relations
                if r["category"] == category
                and r["window"] == window
                and r["kind"] == "negative"
            ]
            if not negatives:
                continue
            columns = ("frequency_a", "frequency_b", "context_mass_a", "context_mass_b")
            candidates = np.array(
                [[np.log1p(float(r[c])) for c in columns] for r in negatives]
            )
            differences = defaultdict(list)
            for p in positive:
                target = np.array([np.log1p(float(p[c])) for c in columns])
                distance = np.abs(candidates - target).sum(axis=1)
                n = negatives[int(np.argmin(distance))]
                record = dict(
                    corpus=corpus,
                    category=category,
                    window=window,
                    positive_pair=p["pair_id"],
                    negative_pair=n["pair_id"],
                    log_frequency_and_context_match_distance=float(distance.min()),
                )
                for column in columns:
                    record["log_difference_" + column] = float(
                        np.log1p(float(p[column])) - np.log1p(float(n[column]))
                    )
                for feature in (
                    "direct_per_occurrence",
                    "unigram_log2_lift",
                    "context_overlap",
                    "shared_distinctive_mass",
                    "relation_pattern_log2_lift",
                ):
                    difference = float(p[feature]) - float(n[feature])
                    differences[feature].append(difference)
                    record["difference_" + feature] = difference
                records.append(record)
            for feature, values in differences.items():
                values = np.array(values)
                resamples = [
                    values[rng.integers(0, len(values), len(values))].mean()
                    for _ in range(500)
                ]
                low, high = np.quantile(resamples, (0.025, 0.975))
                summaries.append(
                    dict(
                        corpus=corpus,
                        category=category,
                        window=window,
                        feature=feature,
                        positive_pairs=len(values),
                        mean_relation_minus_control=float(values.mean()),
                        ci_low=float(low),
                        ci_high=float(high),
                    )
                )
    write_csv(output / "context_mass_matched_negative_pairs.csv", records)
    write_csv(output / "context_mass_matched_negative_effects.csv", summaries)
    return summaries


def sampled_context_examples(corpus, config, stats, tokens, output):
    records = []
    for row, samples in zip(config["targets"], stats["samples"], strict=True):
        for priority, signature, four, dynamic in sorted(samples)[:8]:
            records.append(
                dict(
                    corpus=corpus,
                    word=tokens[row].decode(errors="replace"),
                    row=row,
                    normalized_sentence_digest=signature,
                    selection_priority=priority,
                    fixed4_context=" ".join(
                        tokens[r].decode(errors="replace") for r in four
                    ),
                    dynamic10_context=" ".join(
                        tokens[r].decode(errors="replace") for r in dynamic
                    ),
                    sense_annotation="unlabeled",
                )
            )
    write_csv(output / "sampled_context_examples.csv", records)


def exposure_stability(corpus, config, stats, arrays, words, relations, output):
    totals = np.array(
        [
            stats["shards"].get(str(j), {}).get("lexical_tokens", 0)
            for j in range(len(config["shards"]))
        ],
        dtype=float,
    )
    valid = totals > 0
    index = {r: i for i, r in enumerate(config["targets"])}
    corrected = {}
    for row, i in index.items():
        counts = arrays["shard_words"][valid, i].astype(float)
        rates = counts / totals[valid]
        mean = counts.sum() / totals[valid].sum()
        variance = float(np.average((rates - mean) ** 2, weights=totals[valid]))
        corrected[row] = variance**0.5 / mean if mean else 0
    for r in words:
        r["shard_rate_cv"] = corrected[int(r["row"])]
    token_rows = {r["word"]: int(r["row"]) for r in words}
    for r in relations:
        r["shard_rate_cv"] = (
            corrected[token_rows[r["word_a"]]] + corrected[token_rows[r["word_b"]]]
        ) / 2
    write_csv(output / "word_features.csv", words)
    records = []
    cumsum = totals.cumsum()
    for fraction in (0.25, 0.5, 0.75, 1):
        end = min(len(totals), int(np.searchsorted(cumsum, 320_000_000 * fraction)) + 1)
        amount = float(totals[:end].sum())
        for category in config["groups"]:
            pair_ids = [
                i
                for i, p in enumerate(config["pairs"])
                if p["category"] == category and p["kind"] == "relation"
            ]
            for w, window in enumerate(WINDOWS):
                full = (
                    arrays["shard_pairs"][:, pair_ids, w].sum(axis=0).astype(float)
                    / 320_000_000
                )
                prefix = (
                    arrays["shard_pairs"][:end, pair_ids, w].sum(axis=0).astype(float)
                    / amount
                )
                differences = np.divide(
                    prefix - full, full, out=np.zeros_like(full), where=full > 0
                )
                records.append(
                    dict(
                        corpus=corpus,
                        category=category,
                        window=window,
                        prefix_requested_fraction=fraction,
                        actual_prefix_tokens=int(amount),
                        unique_pairs=len(pair_ids),
                        median_absolute_pair_rate_relative_change=float(
                            np.median(abs(differences))
                        ),
                        pair_zero_exposure_fraction=float(np.mean(prefix == 0)),
                    )
                )
    write_csv(output / "relation_prefix_stability.csv", records)
