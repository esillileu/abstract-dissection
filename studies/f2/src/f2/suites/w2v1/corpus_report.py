"""Corpus-only synthesis, pair-level uncertainty, and reproducible evidence tables."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .corpus_evidence import (
    discriminability_evidence,
    exposure_stability,
    mass_matched_negatives,
    sampled_context_examples,
)
from .corpus_investigation import CORPORA, MAIN_CATEGORIES, csv_rows, truth
from .corpus_statistics import WINDOWS, association, spelling_key
from .diagnostic import MODELS, SEEDS
from .relations import write_csv

FEATURES = (
    "exact_spelling_share",
    "spelling_variant_entropy",
    "dedup_exposure_fraction",
    "normalized_dedup_exposure_fraction",
    "context_diversity_per_occurrence",
    "context_events_per_occurrence",
    "fixed4_compressed_distance_fraction",
    "context_entropy_bits",
    "top10_context_mass",
    "distinctive_context_mass",
    "context_overlap",
    "context_js_bits",
    "shared_distinctive_mass",
    "relation_pattern_log2_lift",
    "repeated_relation_pattern_mass",
    "max_shard_share",
    "shard_rate_cv",
    "unigram_log2_lift",
    "direct_per_occurrence",
    "conditional_proxy_discriminability",
    "single_context_center_information_bits",
)


def table(rows, fields):
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                f"{row[f]:.4f}" if isinstance(row[f], float) else str(row[f])
                for f in fields
            )
            + " |"
        )
    return lines


def accuracy_records(corpus, data, questions, outcomes):
    pair_counts = defaultdict(lambda: [0, 0])
    word_counts = defaultdict(lambda: [0, 0])
    category_counts = defaultdict(lambda: [0, 0, 0])
    common_counts = defaultdict(lambda: [0, 0])
    all_counts = defaultdict(lambda: [0, 0])
    for row in outcomes:
        qid = int(row["question_id"])
        q = questions[qid]
        model = row["model"]
        seed = int(row["seed"])
        section = "syntactic" if q.category.startswith("gram") else "semantic"
        included = truth(row["included"])
        for category in (q.category, section, "total"):
            counts = category_counts[model, seed, category]
            counts[2] += 1
            if included:
                counts[0] += truth(row["correct"])
                counts[1] += 1
                if truth(row["common_subset"]):
                    counts = common_counts[model, seed, category]
                    counts[0] += truth(row["correct"])
                    counts[1] += 1
        if not included:
            continue
        ids = data["mapped"][qid]
        correct = truth(row["correct"])
        # Primary statistical unit: unique directed target relation (c,d).
        target_pair = ids[2:]
        counts = pair_counts[model, seed, q.category, target_pair]
        counts[0] += correct
        counts[1] += 1
        if truth(row["common_subset"]):
            counts = all_counts[model, seed, q.category, target_pair]
            counts[0] += correct
            counts[1] += 1
        for word in set(ids):
            counts = word_counts[model, seed, q.category, word]
            counts[0] += correct
            counts[1] += 1
    category = []
    for (model, seed, name), (correct, included, total) in sorted(
        category_counts.items()
    ):
        ccorrect, cincluded = common_counts[model, seed, name]
        category.append(
            dict(
                corpus=corpus,
                model=model,
                seed=seed,
                category=name,
                total_questions=total,
                included=included,
                excluded=total - included,
                correct=correct,
                accuracy_percent=100 * correct / included if included else 0,
                common_included=cincluded,
                common_accuracy_percent=100 * ccorrect / cincluded if cincluded else 0,
            )
        )
    pairs = []
    for (model, seed, name, (a, b)), (correct, total) in sorted(pair_counts.items()):
        ccorrect, ctotal = all_counts[model, seed, name, (a, b)]
        pairs.append(
            dict(
                corpus=corpus,
                model=model,
                seed=seed,
                category=name,
                a=a,
                b=b,
                word_a=data["tokens"][a].decode(errors="replace"),
                word_b=data["tokens"][b].decode(errors="replace"),
                question_count=total,
                correct=correct,
                accuracy_percent=100 * correct / total,
                common_question_count=ctotal,
                common_accuracy_percent=100 * ccorrect / ctotal if ctotal else None,
            )
        )
    words = [
        dict(
            corpus=corpus,
            model=model,
            seed=seed,
            category=name,
            row=word,
            word=data["tokens"][word].decode(errors="replace"),
            question_count=total,
            accuracy_percent=100 * correct / total,
        )
        for (model, seed, name, word), (correct, total) in sorted(word_counts.items())
    ]
    return category, pairs, words


def bootstrap_difference(differences, rng, repetitions=500):
    if not len(differences):
        return float("nan"), float("nan"), float("nan")
    samples = [
        float(np.mean(differences[rng.integers(0, len(differences), len(differences))]))
        for _ in range(repetitions)
    ]
    return float(np.mean(differences)), *np.quantile(samples, (0.025, 0.975))


def category_characteristics(corpus, config, words, overlaps, proxy, accuracy):
    mapped = {(int(r["row"]), r["window"]): r for r in words}
    controls = {int(k): v for k, v in config["controls"].items()}
    fields = (
        "frequency",
        "exact_spelling_share",
        "retained_spelling_family_share",
        "spelling_variant_entropy",
        "dedup_exposure_fraction",
        "normalized_dedup_exposure_fraction",
        "context_diversity_per_occurrence",
        "context_events_per_occurrence",
        "context_entropy_bits",
        "top10_context_mass",
        "distinctive_context_mass",
        "max_shard_share",
        "shard_rate_cv",
        "fixed4_compressed_distance_fraction",
    )
    rows = []
    matched = []
    rng = np.random.default_rng(20261011)
    for category, ids in config["groups"].items():
        for window in WINDOWS:
            record = dict(
                corpus=corpus, category=category, window=window, unique_words=len(ids)
            )
            for field in fields:
                values = np.array([float(mapped[r, window][field]) for r in ids])
                record[field + "_median"] = float(np.median(values))
                differences = np.array(
                    [
                        float(mapped[r, window][field])
                        - float(mapped[controls[r], window][field])
                        for r in ids
                    ]
                )
                mean, low, high = bootstrap_difference(differences, rng)
                matched.append(
                    dict(
                        corpus=corpus,
                        category=category,
                        window=window,
                        feature=field,
                        unique_matched_words=len(ids),
                        mean_benchmark_minus_control=mean,
                        ci_low=low,
                        ci_high=high,
                    )
                )
            selections = [
                r
                for r in overlaps
                if r["category"] == category and r["window"] == window
            ]
            record["within_category_js_median"] = float(
                np.median([float(r["js_bits"]) for r in selections])
            )
            record["within_category_overlap_median"] = float(
                np.median([float(r["overlap_mass"]) for r in selections])
            )
            selections = [
                r for r in proxy if r["category"] == category and r["window"] == window
            ]
            record["conditional_proxy_macro_accuracy"] = (
                float(np.mean([float(r["heldout_accuracy"]) for r in selections]))
                if selections
                else None
            )
            record["conditional_proxy_chance"] = (
                float(selections[0]["chance_accuracy"]) if selections else None
            )
            controls_proxy = [
                r
                for r in proxy
                if r["category"] == category + "-matched-controls"
                and r["window"] == window
            ]
            record["control_proxy_macro_accuracy"] = (
                float(np.mean([float(r["heldout_accuracy"]) for r in controls_proxy]))
                if controls_proxy
                else None
            )
            for model in MODELS:
                selected = [
                    r
                    for r in accuracy
                    if r["model"] == model and r["category"] == category
                ]
                record[model + "_accuracy_mean"] = float(
                    np.mean([r["accuracy_percent"] for r in selected])
                )
                record[model + "_common_accuracy_mean"] = float(
                    np.mean([r["common_accuracy_percent"] for r in selected])
                )
                record[model + "_accuracy_seed_sd"] = float(
                    np.std([r["accuracy_percent"] for r in selected], ddof=1)
                )
            rows.append(record)
    return rows, matched


def feature_outcome_links(corpus, relation_rows, pair_accuracy):
    joined = []
    for r in relation_rows:
        if r["kind"] != "relation":
            continue
        key = r["category"], r["word_a"], r["word_b"]
        selected = [
            p
            for p in pair_accuracy
            if p["category"] == key[0]
            and p["word_a"] == key[1]
            and p["word_b"] == key[2]
        ]
        a = {
            m: [p["accuracy_percent"] for p in selected if p["model"] == m]
            for m in MODELS
        }
        if not all(len(a[m]) == 3 for m in MODELS):
            continue
        common = {
            m: [
                p["common_accuracy_percent"]
                for p in selected
                if p["model"] == m and p["common_question_count"] > 0
            ]
            for m in MODELS
        }
        has_common = all(len(common[m]) == 3 for m in MODELS)
        joined.append(
            dict(
                r,
                cbow_accuracy=float(np.mean(a["cbow"])),
                skipgram_accuracy=float(np.mean(a["skipgram"])),
                gap=float(np.mean(a["skipgram"]) - np.mean(a["cbow"])),
                cbow_seed_sd=float(np.std(a["cbow"], ddof=1)),
                skipgram_seed_sd=float(np.std(a["skipgram"], ddof=1)),
                common_cbow_accuracy=float(np.mean(common["cbow"]))
                if has_common
                else None,
                common_skipgram_accuracy=float(np.mean(common["skipgram"]))
                if has_common
                else None,
                common_gap=float(np.mean(common["skipgram"]) - np.mean(common["cbow"]))
                if has_common
                else None,
            )
        )
    results = []
    rng = np.random.default_rng(20261011)
    for window in WINDOWS:
        rows = [r for r in joined if r["window"] == window]
        pair_keys = sorted({(r["word_a"], r["word_b"]) for r in rows})
        cluster_ids = {pair: i for i, pair in enumerate(pair_keys)}
        clusters = [cluster_ids[r["word_a"], r["word_b"]] for r in rows]
        categories = sorted({r["category"] for r in rows})
        logf = np.array([np.log1p(float(r["min_frequency"])) for r in rows])
        ratio = np.array(
            [
                abs(
                    np.log(
                        (float(r["frequency_a"]) + 1) / (float(r["frequency_b"]) + 1)
                    )
                )
                for r in rows
            ]
        )
        logf = (logf - logf.mean()) / logf.std()
        base = np.column_stack([np.ones(len(rows)), logf, logf**2, ratio])
        full = np.column_stack(
            [
                base,
                *[
                    np.array([r["category"] == c for r in rows], dtype=float)
                    for c in categories[1:]
                ],
            ]
        )
        for feature in FEATURES:
            values = np.array([float(r[feature]) for r in rows])
            if feature in (
                "context_diversity_per_occurrence",
                "shard_rate_cv",
                "direct_per_occurrence",
            ):
                values = np.log1p(values)
            for outcome in ("cbow_accuracy", "gap"):
                y = np.array([r[outcome] for r in rows])
                for label, controls in (
                    ("unadjusted", np.ones((len(rows), 1))),
                    ("frequency", base),
                    ("frequency_category", full),
                ):
                    coefficient, low, high = association(
                        values, y, controls, rng, repetitions=500, clusters=clusters
                    )
                    results.append(
                        dict(
                            corpus=corpus,
                            window=window,
                            feature=feature,
                            outcome=outcome,
                            adjustment=label,
                            unique_pairs=len(pair_keys),
                            pair_category_rows=len(rows),
                            partial_correlation=coefficient,
                            ci_low=float(low),
                            ci_high=float(high),
                        )
                    )
    return joined, results


def negative_control_effects(corpus, relations):
    rng = np.random.default_rng(20261011)
    result = []
    for category in sorted({r["category"] for r in relations}):
        for window in WINDOWS:
            positives = {
                r["pair_id"]: r
                for r in relations
                if r["category"] == category
                and r["window"] == window
                and r["kind"] == "relation"
            }
            negatives = [
                r
                for r in relations
                if r["category"] == category
                and r["window"] == window
                and r["kind"] == "negative"
            ]
            for feature in (
                "unigram_log2_lift",
                "direct_per_occurrence",
                "context_overlap",
                "context_js_bits",
                "shared_distinctive_mass",
                "relation_pattern_log2_lift",
                "repeated_relation_pattern_mass",
            ):
                differences = np.array(
                    [
                        float(positives[n["matched_relation"]][feature])
                        - float(n[feature])
                        for n in negatives
                    ]
                )
                mean, low, high = bootstrap_difference(differences, rng)
                result.append(
                    dict(
                        corpus=corpus,
                        category=category,
                        window=window,
                        feature=feature,
                        matched_pairs=len(differences),
                        mean_relation_minus_control=mean,
                        ci_low=float(low),
                        ci_high=float(high),
                    )
                )
    return result


def spelling_coverage(corpus, data, questions, stats):
    variants = defaultdict(list)
    for t, n in stats["raw_variants"]:
        variants[spelling_key(t.encode())].append((t, n))
    rows = []
    for category in sorted({q.category for q in questions}):
        words = sorted(
            {
                t
                for q in questions
                if q.category == category
                for t in (q.a, q.b, q.c, q.expected)
            }
        )
        eval_rows = {t.upper(): data["lookup"].row(t) for t in data["tokens"][:82000]}
        # Use the evaluator's canonical row selection rather than a spelling guess.
        from f2.suites.w2v.evaluation import _analogy_rows

        eval_rows = _analogy_rows(data["lookup"], 82000)
        for word in words:
            row = eval_rows.get(word.upper())
            v = variants[spelling_key(word)]
            total = sum(c for _, c in v)
            rows.append(
                dict(
                    corpus=corpus,
                    category=category,
                    benchmark_word=word.decode(),
                    included=row is not None,
                    training_word=data["tokens"][row].decode(errors="replace")
                    if row is not None
                    else "",
                    training_frequency=int(data["lookup"].counts[row])
                    if row is not None
                    else 0,
                    raw_exact_benchmark_spelling_frequency=sum(
                        n for t, n in v if t.encode() == word
                    ),
                    raw_orthographic_family_frequency=total,
                    orthographic_forms=len(v),
                    semantic_alias_or_sense_annotation="unavailable",
                )
            )
    return rows


def distribution_records(corpus, stats, config):
    total = stats["total"]
    lengths = {int(k): int(v) for k, v in stats["raw_lengths"].items()}
    n = sum(lengths.values())
    ordered = sorted(lengths)
    cumulative = np.cumsum([lengths[x] for x in ordered])
    record = dict(
        corpus=corpus,
        lexical_tokens=total["lexical_tokens"],
        raw_lines=total["lines"],
        training_blocks=total["training_blocks"],
        mean_raw_line_length=sum(k * v for k, v in lengths.items()) / n,
        median_raw_line_length=ordered[int(np.searchsorted(cumulative, 0.5 * n))],
        p95_raw_line_length=ordered[int(np.searchsorted(cumulative, 0.95 * n))],
        oov_token_fraction=total["oov_tokens"] / total["lexical_tokens"],
        exact_duplicate_line_fraction=total["duplicate_lines"] / total["lines"],
        normalized_duplicate_line_fraction=total["normalized_duplicate_lines"]
        / total["lines"],
        oov_bridged_adjacency_fraction=total["oov_bridged_adjacencies"]
        / total["filtered_adjacencies"],
        truncated_tokens=total.get("truncated_tokens", 0),
        nul_tokens=total.get("nul_tokens", 0),
        materialized_sha256=stats["sha256"],
        consumed_shards=len(stats["shards"]),
    )
    return record


def stability_records(corpus, data, config, stats, arrays):
    # Prefixes use shard token totals, never treat shards as articles/documents.
    totals = np.array(
        [
            stats["shards"].get(str(j), {}).get("lexical_tokens", 0)
            for j in range(len(config["shards"]))
        ],
        dtype=float,
    )
    cumsum = totals.cumsum()
    records = []
    for fraction in (0.25, 0.5, 0.75, 1):
        end = min(len(totals), int(np.searchsorted(cumsum, 320_000_000 * fraction)) + 1)
        tokens = float(totals[:end].sum())
        words = arrays["shard_words"][:end].sum(axis=0)
        for category, ids in config["groups"].items():
            positions = [config["targets"].index(r) for r in ids]
            full = (
                arrays["shard_words"][:, positions].sum(axis=0).astype(float)
                / 320_000_000
            )
            current = words[positions].astype(float) / tokens
            relative = np.divide(
                current - full, full, out=np.zeros_like(full), where=full > 0
            )
            records.append(
                dict(
                    corpus=corpus,
                    category=category,
                    prefix_requested_fraction=fraction,
                    consumed_shards=end,
                    actual_prefix_tokens=int(tokens),
                    median_absolute_word_rate_relative_change=float(
                        np.median(abs(relative))
                    ),
                    word_rate_correlation=float(np.corrcoef(current, full)[0, 1]),
                    coverage_fraction=float(np.mean(current > 0)),
                )
            )
    return records


def word_outcome_links(corpus, words, word_accuracy, discrimination):
    scores = defaultdict(lambda: defaultdict(list))
    for r in word_accuracy:
        scores[r["category"], r["row"]][r["model"]].append(r["accuracy_percent"])
    records = []
    byrow = {(int(r["row"]), r["window"]): r for r in words}
    context_lookup = {
        (r["category"], r["window"], int(r["row"])): r for r in discrimination
    }
    for (category, row), models in scores.items():
        if not all(len(models[m]) == 3 for m in MODELS):
            continue
        for window in WINDOWS:
            r = byrow[row, window]
            record = dict(
                r,
                category=category,
                cbow_accuracy=float(np.mean(models["cbow"])),
                skipgram_accuracy=float(np.mean(models["skipgram"])),
                gap=float(np.mean(models["skipgram"]) - np.mean(models["cbow"])),
            )
            for feature in (
                "conditional_proxy_discriminability",
                "single_context_center_information_bits",
            ):
                record[feature] = context_lookup[category, window, row][feature]
            records.append(record)
    results = []
    rng = np.random.default_rng(20261011)
    for window in WINDOWS:
        # One record per unique word across categories; mean over its category scores.
        grouped = defaultdict(list)
        for r in records:
            if r["window"] == window:
                grouped[int(r["row"])].append(r)
        rows = []
        for _, items in sorted(grouped.items()):
            r = dict(items[0])
            r["cbow_accuracy"] = float(np.mean([x["cbow_accuracy"] for x in items]))
            r["gap"] = float(np.mean([x["gap"] for x in items]))
            r["category_memberships"] = {x["category"] for x in items}
            for feature in (
                "conditional_proxy_discriminability",
                "single_context_center_information_bits",
            ):
                r[feature] = float(np.mean([x[feature] for x in items]))
            rows.append(r)
        categories = sorted({c for r in rows for c in r["category_memberships"]})
        logf = np.log1p([float(r["frequency"]) for r in rows])
        logf = (logf - logf.mean()) / logf.std()
        controls = np.column_stack(
            [
                np.ones(len(rows)),
                logf,
                logf**2,
                *[
                    np.array(
                        [c in r["category_memberships"] for r in rows], dtype=float
                    )
                    for c in categories[:-1]
                ],
            ]
        )
        for feature in (
            *FEATURES[:10],
            "max_shard_share",
            "shard_rate_cv",
            "conditional_proxy_discriminability",
            "single_context_center_information_bits",
        ):
            x = np.array([float(r[feature]) for r in rows])
            for outcome in ("cbow_accuracy", "gap"):
                observed, low, high = association(
                    x,
                    np.array([r[outcome] for r in rows]),
                    controls,
                    rng,
                    repetitions=500,
                )
                results.append(
                    dict(
                        corpus=corpus,
                        window=window,
                        feature=feature,
                        outcome=outcome,
                        adjustment="frequency_category_memberships",
                        unique_words=len(rows),
                        partial_correlation=observed,
                        ci_low=float(low),
                        ci_high=float(high),
                    )
                )
    return records, results


def build_report(inputs, questions, common, caches, output):
    categories = []
    distribution = []
    associations = []
    word_associations = []
    negative = []
    characteristics = []
    matched = []
    stability = []
    manifest = dict(
        analysis="W2V1 Table3 corpus-only investigation",
        corpora=list(CORPORA),
        seeds=list(SEEDS),
        common_questions=len(common),
        methods=dict(
            exact="whole 320M materialized stream counts, hashes, spelling forms, sentence/shard exposure, directional context distributions, direct pairs and template counts",
            approximations="KMV-256 unique context estimate; up to 128 bottom-hash unique normalized-template/position samples per word for grouped holdout conditional prediction; normalized template duplicate proxy; nearby-frequency within-category word-pair sampling",
            inference="500 unique-pair/unique-word percentile bootstrap replicates; exploratory partial correlations; log frequency, squared log frequency, pair frequency imbalance and category controls",
            window="CBOW fixed4; SG radius uniformly 1..10 expectation (11-d)/10; newlines, literal EOS and 1000 retained-token blocks; no subsampling; no actual worker byte offsets/RNG replay",
            limitations="No LDC original corpus, no article metadata, no semantic alias/sense annotations, no learned geometry/neighbor/optimization/implementation analysis",
        ),
    )
    for corpus in CORPORA:
        cache = caches[corpus]
        out = output / corpus
        config = json.loads((cache / "configuration.json").read_text())
        stats = json.loads((cache / "statistics.json").read_text())
        arrays = np.load(cache / "statistics.npz")
        verification = json.loads((cache / "verification.json").read_text())
        if (
            not verification["sha256_matches"]
            or verification["vocabulary_count_mismatches"]
        ):
            raise ValueError("Unverified corpus cannot be reported")
        (out / "configuration.json").write_text(json.dumps(config, indent=2))
        (out / "verification.json").write_text(json.dumps(verification, indent=2))
        outcomes = csv_rows(cache / "question_outcomes.csv")
        category, pairs, wordscores = accuracy_records(
            corpus, inputs[corpus], questions, outcomes
        )
        categories.extend(category)
        write_csv(out / "category_accuracy_by_seed.csv", category)
        write_csv(out / "pair_accuracy_by_seed.csv", pairs)
        write_csv(out / "word_accuracy_by_seed.csv", wordscores)
        words = csv_rows(out / "word_features.csv")
        relations = csv_rows(out / "relation_features.csv")
        # Convert typed identifiers for relational joins.
        for r in relations:
            for key in ("pair_id", "matched_relation"):
                r[key] = int(r[key])
        proxy = csv_rows(out / "conditional_prediction_proxy.csv")
        exposure_stability(corpus, config, stats, arrays, words, relations, out)
        discrimination = discriminability_evidence(
            corpus, config, cache, out, relations, words, proxy
        )
        mass_matched_negatives(corpus, relations, out)
        sampled_context_examples(corpus, config, stats, inputs[corpus]["tokens"], out)
        joined, links = feature_outcome_links(corpus, relations, pairs)
        associations.extend(links)
        write_csv(out / "pair_features_with_accuracy.csv", joined)
        write_csv(out / "pair_feature_associations.csv", links)
        joinedwords, wordlinks = word_outcome_links(
            corpus, words, wordscores, discrimination
        )
        word_associations.extend(wordlinks)
        write_csv(out / "word_features_with_accuracy.csv", joinedwords)
        write_csv(out / "word_feature_associations.csv", wordlinks)
        overlaps = csv_rows(out / "context_overlap_comparisons.csv")
        char, controls = category_characteristics(
            corpus, config, words, overlaps, proxy, category
        )
        characteristics.extend(char)
        matched.extend(controls)
        negative.extend(negative_control_effects(corpus, relations))
        distribution.append(distribution_records(corpus, stats, config))
        stability.extend(
            stability_records(corpus, inputs[corpus], config, stats, arrays)
        )
        write_csv(
            out / "benchmark_word_coverage.csv",
            spelling_coverage(corpus, inputs[corpus], questions, stats),
        )
        manifest[corpus] = dict(
            corpus_sha256=verification["corpus_sha256"],
            vocabulary_digest=config["vocabulary_digest"],
            vocabulary_counts_verified=verification["vocabulary_rows"],
            cache_directory=str(cache),
            context_parquet=str(cache / "context_counts.parquet"),
            analysis_seconds=verification["elapsed_seconds"],
            runs=config["runs"],
        )
    write_csv(output / "corpus_distribution.csv", distribution)
    write_csv(output / "category_characteristics.csv", characteristics)
    write_csv(output / "matched_word_feature_differences.csv", matched)
    write_csv(output / "negative_relation_controls.csv", negative)
    write_csv(output / "prefix_stability.csv", stability)
    write_csv(output / "pair_feature_associations.csv", associations)
    write_csv(output / "word_feature_associations.csv", word_associations)
    cross = []
    for corpus in CORPORA:
        for category in ("semantic", "syntactic", "total", *MAIN_CATEGORIES):
            r = dict(corpus=corpus, category=category)
            for model in MODELS:
                scores = [
                    x
                    for x in categories
                    if x["corpus"] == corpus
                    and x["category"] == category
                    and x["model"] == model
                ]
                r["full_" + model] = float(
                    np.mean([x["accuracy_percent"] for x in scores])
                )
                r["common_" + model] = float(
                    np.mean([x["common_accuracy_percent"] for x in scores])
                )
                r["common_" + model + "_seed_sd"] = float(
                    np.std([x["common_accuracy_percent"] for x in scores], ddof=1)
                )
                r["full_included"] = scores[0]["included"]
                r["common_included"] = scores[0]["common_included"]
            r["common_sg_minus_cbow"] = r["common_skipgram"] - r["common_cbow"]
            cross.append(r)
    write_csv(output / "cross_corpus_accuracy.csv", cross)
    # Paired changes on identical directed pairs, not question-level resampling.
    crosslinks = []
    rng = np.random.default_rng(20261011)
    records = {
        c: csv_rows(output / c / "pair_features_with_accuracy.csv") for c in CORPORA
    }
    for other in ("wmt", "umbc"):
        for window in WINDOWS:
            left = {
                (r["category"], r["word_a"], r["word_b"]): r
                for r in records["lm1b"]
                if r["window"] == window and r["common_gap"] != ""
            }
            right = {
                (r["category"], r["word_a"], r["word_b"]): r
                for r in records[other]
                if r["window"] == window and r["common_gap"] != ""
            }
            keys = sorted(left.keys() & right.keys())
            endpoints = sorted({k[1:] for k in keys})
            endpoint_ids = {pair: i for i, pair in enumerate(endpoints)}
            clusters = [endpoint_ids[k[1:]] for k in keys]
            y = np.array(
                [
                    float(left[k]["common_gap"]) - float(right[k]["common_gap"])
                    for k in keys
                ]
            )
            lf = np.array(
                [
                    np.log1p(float(left[k]["min_frequency"]))
                    - np.log1p(float(right[k]["min_frequency"]))
                    for k in keys
                ]
            )
            cats = sorted({k[0] for k in keys})
            controls = np.column_stack(
                [
                    np.ones(len(keys)),
                    lf,
                    *[
                        np.array([k[0] == c for k in keys], dtype=float)
                        for c in cats[1:]
                    ],
                ]
            )
            for feature in FEATURES:
                x = np.array(
                    [float(left[k][feature]) - float(right[k][feature]) for k in keys]
                )
                coefficient, low, high = association(
                    x, y, controls, rng, repetitions=500, clusters=clusters
                )
                crosslinks.append(
                    dict(
                        comparison="lm1b-minus-" + other,
                        window=window,
                        feature=feature,
                        unique_common_pairs=len(endpoints),
                        frequency_category_controlled_correlation_with_gap_change=coefficient,
                        ci_low=float(low),
                        ci_high=float(high),
                    )
                )
    write_csv(output / "cross_corpus_pair_associations.csv", crosslinks)
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    render_report(
        output,
        distribution,
        characteristics,
        cross,
        associations,
        word_associations,
        negative,
        matched,
        crosslinks,
        manifest,
    )


def render_report(
    output,
    distribution,
    characteristics,
    cross,
    associations,
    word_associations,
    negative,
    matched,
    crosslinks,
    manifest,
):
    lines = [
        "# W2V1 Table 3 corpus-centric investigation",
        "",
        "## Observations and scope",
        "",
        "All three completed corpora, both models and seeds 1/7/19 are included. Each materialized 320M stream is scanned once; all 82,001 vocabulary rows (82K lexical plus EOS) are compared to the saved token counts. Analogy outcomes use the recorded 82,000-row evaluation boundary, with the same candidate vocabulary as the original evaluation. The common question subset restricts questions only. No retraining or learned geometry, nearest-neighbor, optimization, or implementation diagnosis is performed.",
        "",
        "LDC original training data is not available: no direct comparison to the paper's original corpus is possible. WMT and LM1B may share news sources and are not independent corpus samples. Source normalization before the stored materialized stream cannot be reversed. Surface aliases are measured only by explicit case/punctuation/hyphen/underscore/camel-case rules; semantic aliases and word senses are not labeled.",
        "",
        "## A. Corpus characteristics comparison",
        "",
        "### Corpus distribution and coverage",
        "",
    ]
    lines.extend(
        table(
            distribution,
            [
                "corpus",
                "lexical_tokens",
                "raw_lines",
                "mean_raw_line_length",
                "oov_token_fraction",
                "exact_duplicate_line_fraction",
                "normalized_duplicate_line_fraction",
                "oov_bridged_adjacency_fraction",
                "consumed_shards",
            ],
        )
    )
    lines.extend(
        [
            "",
            "These are raw line instances and retained-token training blocks, not documents. Exact duplicates use a 128-bit digest (negligible collision probability). “Normalized duplicate” means the declared normalization heuristic matches; it is a template redundancy proxy, not a comprehensive near-duplicate detector. Vocabulary loss and OOV-created adjacency are distinct quantities.",
            "",
            "### Category comparison (fixed4; unique words have equal weight)",
            "",
        ]
    )
    selected = [
        r
        for r in characteristics
        if r["window"] == "fixed4" and r["category"] in MAIN_CATEGORIES
    ]
    lines.extend(
        table(
            selected,
            [
                "corpus",
                "category",
                "unique_words",
                "frequency_median",
                "exact_spelling_share_median",
                "dedup_exposure_fraction_median",
                "context_diversity_per_occurrence_median",
                "context_entropy_bits_median",
                "within_category_js_median",
                "conditional_proxy_macro_accuracy",
                "conditional_proxy_chance",
                "max_shard_share_median",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Full grammatical category controls and dynamic10 rows are in `category_characteristics.csv`; word-level values and frequency/case/length matching diagnostics are retained. Medians describe different word sets and should not be interpreted as frequency-adjusted effects.",
            "",
            "### Identical evaluation subset across corpora",
            "",
        ]
    )
    lines.extend(
        table(
            cross,
            [
                "corpus",
                "category",
                "full_included",
                "common_included",
                "common_cbow",
                "common_skipgram",
                "common_sg_minus_cbow",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Arithmetic seed means; seed SD is in the CSV, separate from word/pair bootstrap uncertainty.",
            "",
            "### Frequency- and category-controlled pair associations",
            "",
        ]
    )
    candidates = [
        r
        for r in associations
        if r["corpus"] == "lm1b"
        and r["window"] == "fixed4"
        and r["adjustment"] == "frequency_category"
    ]
    ordered = sorted(candidates, key=lambda r: -abs(r["partial_correlation"]))
    lines.extend(
        table(
            ordered,
            [
                "feature",
                "outcome",
                "unique_pairs",
                "partial_correlation",
                "ci_low",
                "ci_high",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Positive gap association means the feature is larger where SG-CBOW accuracy is larger. These exploratory associations are not causal and no question-independent significance tests are used. Percentile intervals resample unique directed target pairs, refitting controls each time; pairs sharing endpoints and shared source questions remain dependent. Word-level sensitivity is separately tabulated. No multiple-testing claim is made.",
            "",
            "### Negative relation controls and window exposure",
            "",
        ]
    )
    selected = [
        r
        for r in negative
        if r["corpus"] == "lm1b"
        and r["category"] in MAIN_CATEGORIES
        and r["feature"]
        in (
            "relation_pattern_log2_lift",
            "shared_distinctive_mass",
            "direct_per_occurrence",
        )
    ]
    lines.extend(
        table(
            selected,
            [
                "category",
                "window",
                "feature",
                "matched_pairs",
                "mean_relation_minus_control",
                "ci_low",
                "ci_high",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Negative pairs keep the first endpoint and match the second endpoint's frequency within the category. They need not be semantically unrelated in every sense. Their context amounts are measured and may differ; these are endpoint-frequency-matched controls, with a context-mass-matched sensitivity supplied below. Generic versus distinctive shared context uses corpus unigram lift ≥4; this does not label a sentence as stating a capital/state relation. Patterns retain frequent intervening lexical words, direction, and `<WORD>` for other tokens. Leave-pair-out lift excludes the positive and its matched negative from pattern fitting. Patterns count direct cooccurrence instances, not asserted relations.",
            "",
            "### Cross-corpus feature changes and gap changes",
            "",
        ]
    )
    selected = sorted(
        crosslinks,
        key=lambda r: (
            -abs(r["frequency_category_controlled_correlation_with_gap_change"])
        ),
    )[:20]
    lines.extend(
        table(
            selected,
            [
                "comparison",
                "window",
                "feature",
                "unique_common_pairs",
                "frequency_category_controlled_correlation_with_gap_change",
                "ci_low",
                "ci_high",
            ],
        )
    )
    lines.extend(
        [
            "",
            "Both identical target pairs and identical source questions are used for these feature-change comparisons. Cross-corpus causal conclusions remain limited by corpus overlap and three observed corpus conditions.",
            "",
            "## B. Hypothesis evaluation",
            "",
        ]
    )
    # Summaries are evidence rules, not a post-hoc search for one favored feature.
    hypothesis_groups = {
        "Coverage / normalization": (
            "exact_spelling_share",
            "spelling_variant_entropy",
            "fixed4_compressed_distance_fraction",
        ),
        "Effective exposure shortage": (
            "dedup_exposure_fraction",
            "normalized_dedup_exposure_fraction",
            "context_diversity_per_occurrence",
            "context_events_per_occurrence",
        ),
        "Context ambiguity": (
            "context_entropy_bits",
            "top10_context_mass",
            "distinctive_context_mass",
            "context_overlap",
            "context_js_bits",
            "conditional_proxy_discriminability",
            "single_context_center_information_bits",
        ),
        "Insufficient relation-specific information": (
            "shared_distinctive_mass",
            "relation_pattern_log2_lift",
            "repeated_relation_pattern_mass",
            "unigram_log2_lift",
            "direct_per_occurrence",
        ),
        "Corpus distribution bias": ("max_shard_share", "shard_rate_cv"),
    }
    expected_gap_direction = {
        "exact_spelling_share": -1,
        "spelling_variant_entropy": 1,
        "fixed4_compressed_distance_fraction": 1,
        "dedup_exposure_fraction": -1,
        "normalized_dedup_exposure_fraction": -1,
        "context_diversity_per_occurrence": -1,
        "context_events_per_occurrence": -1,
        "distinctive_context_mass": -1,
        "context_overlap": 1,
        "context_js_bits": -1,
        "conditional_proxy_discriminability": -1,
        "single_context_center_information_bits": -1,
        "shared_distinctive_mass": -1,
        "relation_pattern_log2_lift": -1,
        "repeated_relation_pattern_mass": -1,
        "unigram_log2_lift": -1,
        "direct_per_occurrence": -1,
        "max_shard_share": 1,
        "shard_rate_cv": 1,
    }
    assessments = []
    for hypothesis, features in hypothesis_groups.items():
        links = [
            r for r in candidates if r["outcome"] == "gap" and r["feature"] in features
        ]
        nonzero = [r for r in links if r["ci_low"] > 0 or r["ci_high"] < 0]
        stable = [
            r
            for r in nonzero
            if np.sign(r["partial_correlation"])
            == expected_gap_direction.get(r["feature"], 0)
        ]
        contrary = [
            r
            for r in nonzero
            if r["feature"] in expected_gap_direction and r not in stable
        ]
        best = max(links, key=lambda r: abs(r["partial_correlation"]))
        crossconsistent = []
        for link in stable:
            others = [
                r
                for r in associations
                if r["corpus"] != "lm1b"
                and r["window"] == "fixed4"
                and r["adjustment"] == "frequency_category"
                and r["feature"] == link["feature"]
                and r["outcome"] == "gap"
            ]
            if all(
                np.sign(r["partial_correlation"])
                == np.sign(link["partial_correlation"])
                for r in others
            ):
                crossconsistent.append(link["feature"])
        evidence = (
            "; ".join(
                f"{r['feature']}: r={r['partial_correlation']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                for r in stable
            )
            or "No directional controlled gap association supports this hypothesis with a pair-bootstrap interval excluding zero."
        )
        counter = (
            "; ".join(
                f"{r['feature']}: interval includes zero or direction is not prespecified"
                for r in links
                if r not in nonzero or r["feature"] not in expected_gap_direction
            )
            + "; "
            + "; ".join(
                f"{r['feature']}: opposite to hypothesis, r={r['partial_correlation']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
                for r in contrary
            )
            or "Association direction does not establish an adverse mechanism."
        )
        confound = "Frequency/category dependence; shared endpoints; exploratory feature selection."
        if hypothesis == "Coverage / normalization":
            confound += " Same vocabulary and included questions for both models; no semantic alias/sense annotation."
        elif hypothesis == "Context ambiguity":
            confound += " Entropy, overlap and center-classification proxy measure different properties; proxy is sampled NB, not CBOW."
        elif hypothesis == "Insufficient relation-specific information":
            confound += " Frequent-token templates and unigram lift are not semantic relation annotations."
        elif hypothesis == "Corpus distribution bias":
            confound += " Shards are acquisition partitions, not documents."
        assessment = "Descriptive association; corpus-only explanation not established."
        if crossconsistent:
            assessment += (
                " Same coefficient sign in all three corpora: "
                + ", ".join(crossconsistent)
                + "."
            )
        assessments.append(
            dict(
                hypothesis=hypothesis,
                support=evidence,
                counterevidence=counter,
                confounders=confound,
                judgment=assessment,
                ranking_score=max(
                    (abs(r["partial_correlation"]) for r in stable), default=0
                ),
                strongest_feature=best["feature"],
            )
        )
    assessments.sort(key=lambda r: -r["ranking_score"])
    write_csv(output / "hypothesis_evaluation.csv", assessments)
    lines.extend(
        table(
            assessments,
            ["hypothesis", "support", "counterevidence", "confounders", "judgment"],
        )
    )
    lines.extend(["", "## C. Conclusions", ""])
    strongest = next((r for r in ordered if r["outcome"] == "gap"), ordered[0])
    lines.append(
        f"1. Largest examined LM1B fixed4 frequency/category-controlled gap association: `{strongest['feature']}`, partial r={strongest['partial_correlation']:+.3f}, pair-bootstrap 95% interval [{strongest['ci_low']:+.3f}, {strongest['ci_high']:+.3f}]. This ranks an association, not a root cause."
    )
    lines.append(
        "2. Frequency is controlled using log minimum pair frequency, its square, endpoint frequency imbalance and category indicators. Remaining associations are distinguishable from this specified frequency model, but residual frequency confounding and error-in-variables remain possible. Unadjusted/frequency-only/fully adjusted estimates and unique-word sensitivity are provided."
    )
    lines.append(
        "3. Corpus-side objective distinction: CBOW receives one center-prediction context per retained occurrence (unless the context is empty); SG receives expected individual prediction events weighted by (11-d)/10. Generic/shared surrounding bags can be less discriminative for center classification, whereas individual pair predictions provide separate exposure events. This is a possible objective-level interpretation; no actual loss, gradient or optimization behavior is measured."
    )
    lines.append(
        "4. WMT, LM1B and UMBC have completed matched model/seed sets. Common-question accuracies and pairwise corpus-feature changes are reported above. Three overlapping or heterogeneous corpora do not identify causality; feature direction consistency is explicitly retained in the hypothesis table."
    )
    lines.append(
        "5. Surface coverage, effective context exposure, redundancy, ambiguity proxies and relation templates can quantify corpus-side associations. They cannot establish that these characteristics cause the CBOW deficit relative to the paper, particularly without the original LDC corpus, semantic sense labels or an intervention. **corpus-only explanation insufficient**."
    )
    lines.extend(
        [
            "",
            "Hypotheses are ordered by the largest absolute controlled LM1B gap correlation with an interval excluding zero; this exploratory ranking is not evidence strength after multiplicity correction and is not a ranking of established causes. Counterevidence and cross-corpus directions are included.",
            "",
            "## Measurement and reproducibility limits",
            "",
            "- The exact full-stream inventory is distinguished from approximations in `analysis_manifest.json`. KMV estimates are exact below 256 signatures and approximate above (typical relative sampling error about 1/√254 ≈6.3%; sketch uncertainty is not propagated into the association bootstrap).",
            "- The conditional proxy uses up to 128 bottom-hash unique normalized-template/position samples per word, sentence-normalization-group holdout, uniform center priors and additive smoothing. This sample is not occurrence-weighted. Radius draws are deterministic diagnostics, not actual SG RNG replay. Its balanced macro accuracy is not CBOW accuracy or loss.",
            "- Nonbenchmark controls match frequency, initial capitalization and spelling length; they are not independently annotated as common nouns or non-proper names. Category-wide center discrimination mixes relation endpoints; it does not by itself measure discrimination among cities alone.",
            "- Duplicate line counts and normalized templates are not article duplicates. No document boundaries, article metadata, full near-duplicate semantic matching, semantic alias disambiguation or sense-tagging are available.",
            "- Runtime worker byte-offset starts may introduce extra boundaries; these are not replayed. Statistics describe full corpus sentence-local exposure and SG expected windows, not exact per-seed training event traces.",
            "- Background unigram distributions and count-based features use training counts. Directional context counts are saved losslessly as Parquet in the reconstructible analysis cache, with complete token-row CSV and manifests in deliverables. Full raw questions/pair/word seed outputs permit recomputation and alternate thresholds.",
            "- Statistical intervals use pair/word resampling, never independent-question assumptions. Shared words and paired analogy sources mean these intervals remain exploratory.",
            "",
            "## Verified identities",
            "",
        ]
    )
    for corpus in CORPORA:
        lines.append(
            f"- {corpus}: SHA-256 `{manifest[corpus]['corpus_sha256']}`; {manifest[corpus]['vocabulary_counts_verified']:,} vocabulary-row counts verified; full stream analysis {manifest[corpus]['analysis_seconds']:.1f}s."
        )
    lines.append("")
    (output / "corpus_investigation.md").write_text("\n".join(lines), encoding="utf-8")
