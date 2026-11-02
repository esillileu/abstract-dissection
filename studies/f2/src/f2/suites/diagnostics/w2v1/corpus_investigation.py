"""Integrated, read-only Table 3 corpus investigation over completed runs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from repro_io.checksum import sha256_file
from repro_io.s3 import S3Config, S3ObjectStore
from threadpoolctl import threadpool_limits

from f2.common.paths import get_benchmark_data_dir
from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.evaluation import (
    _analogy_predictions,
    _analogy_rows,
    _analogy_tokens,
    parse_analogy_questions,
)
from repro_core.context import RuntimePaths

from .corpus_statistics import (
    TOKEN_PATTERN,
    WINDOWS,
    CorpusStatistics,
    entropy,
    hash64,
    js_divergence,
    probability,
    spelling_key,
)
from .diagnostic import MODELS, SEEDS
from .relations import corpus_lines, decode_tokens, write_csv

CORPORA = ("lm1b", "wmt", "umbc")
MAIN_CATEGORIES = (
    "capital-world",
    "city-in-state",
    "capital-common-countries",
    "family",
)
QUESTION_SHA256 = "8c29b3332afc46f3fb8be04cb5297bf96f39aa7131272dff57869b4485b22a36"


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def lookup_path(paths, run_id):
    files = list(
        (paths.cache_root / "mlflow_artifact").glob(f"*/{run_id}/lookup/manifest.json")
    )
    if len(files) != 1:
        raise ValueError(f"Expected one cached lookup: {run_id}")
    return files[0].parent


def inventory(paths, questions):
    inputs = {}
    for corpus in CORPORA:
        runs = [
            r
            for r in csv_rows(
                paths.analysis_output("f2", "w2v1") / corpus / "table3-runs.csv"
            )
            if r["model_kind"] in MODELS
        ]
        if {(r["model_kind"], int(r["seed"])) for r in runs} != {
            (m, s) for m in MODELS for s in SEEDS
        }:
            raise ValueError(f"Incomplete Table 3 seeds: {corpus}")
        lookups = [
            load_lookup_artifact(lookup_path(paths, r["mlflow_run_id"])) for r in runs
        ]
        sample = lookups[0]
        for lookup in lookups:
            if lookup.manifest["completed_epochs"] != 3:
                raise ValueError("Incomplete training")
            if lookup.manifest["corpus_digest"] != sample.manifest["corpus_digest"]:
                raise ValueError("Mixed corpus identity")
            for name in ("counts", "token_bytes", "token_offsets"):
                if not np.array_equal(getattr(sample, name), getattr(lookup, name)):
                    raise ValueError("Model/seed vocabulary differs")
        wanted = {token.upper() for q in questions for token in _analogy_tokens(q)}
        rows = _analogy_rows(sample, 82000, wanted=wanted)
        mapped = [
            tuple(rows.get(t.upper()) for t in _analogy_tokens(q)) for q in questions
        ]
        inputs[corpus] = dict(
            runs=runs, lookup=sample, mapped=mapped, tokens=decode_tokens(sample)
        )
    common = [
        i
        for i in range(len(questions))
        if all(all(r is not None for r in inputs[c]["mapped"][i]) for c in CORPORA)
    ]
    return inputs, common


def matched_controls(tokens, counts, targets):
    available = np.array(
        [
            r
            for r, t in enumerate(tokens)
            if r and r not in targets and t.isalpha() and len(t) > 1
        ],
        dtype=np.int64,
    )
    active = np.ones(len(available), dtype=bool)
    capitals = np.array([tokens[r][:1].isupper() for r in available])
    lengths = np.array([len(tokens[r]) for r in available])
    log_counts = np.log(np.asarray(counts[available], dtype=float) + 1)
    controls = {}
    for row in sorted(targets, key=lambda r: (-int(counts[r]), r)):
        token = tokens[row]
        capital = token[:1].isupper()
        allowed = active & (capitals == capital)
        if not allowed.any():
            allowed = active
        distances = np.abs(log_counts - np.log(int(counts[row]) + 1)) + 0.08 * np.abs(
            lengths - len(token)
        )
        distances[~allowed] = np.inf
        position = int(np.argmin(distances))
        match = int(available[position])
        controls[row] = match
        active[position] = False
    return controls


def category_words(questions, mapped):
    groups = defaultdict(set)
    pairs = defaultdict(set)
    for q, ids in zip(questions, mapped, strict=True):
        for row in ids:
            if row is not None:
                groups[q.category].add(row)
        if all(r is not None for r in ids):
            pairs[q.category].update((ids[:2], ids[2:]))
    return groups, pairs


def pair_inventory(groups, counts):
    records = []
    for category, pairs in groups.items():
        forbidden = {frozenset(pair) for pair in pairs}
        right_words = sorted({b for _, b in pairs})
        for a, b in sorted(pairs):
            index = len(records)
            records.append(
                dict(
                    category=category, a=a, b=b, kind="relation", matched_relation=index
                )
            )
            candidates = [
                r for r in right_words if r != a and frozenset((a, r)) not in forbidden
            ]
            if not candidates:
                continue
            negative = min(
                candidates,
                key=lambda r: (
                    abs(np.log((int(counts[r]) + 1) / (int(counts[b]) + 1))),
                    r,
                ),
            )
            records.append(
                dict(
                    category=category,
                    a=a,
                    b=negative,
                    kind="negative",
                    matched_relation=index,
                )
            )
    return records


def bound_lines(shards, store, staging, budget=320_000_000):
    """Annotate materializer-identical bytes with source shard, storing one object."""
    tokens = 0
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="corpus-", dir=staging) as tmp:
        source = Path(tmp) / "shard.txt.zst"
        for shard in shards:
            index = int(shard["shard_index"])
            print(f"Shard {index}: {tokens:,}/{budget:,} lexical tokens", flush=True)
            store.get_file(shard["s3_uri"], source)
            if (
                source.stat().st_size != int(shard["byte_size"])
                or sha256_file(source) != shard["sha256"]
            ):
                raise ValueError("Shard checksum mismatch")
            for line in corpus_lines(source):
                matches = list(TOKEN_PATTERN.finditer(line))
                available = budget - tokens
                if len(matches) <= available:
                    yield index, line
                    tokens += len(matches)
                else:
                    if available:
                        yield (
                            index,
                            line[: matches[available - 1].end()].rstrip(b"\r\n")
                            + b"\n",
                        )
                        tokens += available
                    break
            source.unlink()
            if tokens == budget:
                return
    raise ValueError(f"Corpus budget not met: {tokens}")


def store_from_environment(paths):
    from dotenv import dotenv_values

    settings = dotenv_values(paths.repo_root / ".env")

    def value(suffix):
        key = "F2_CORPUS_S3_" + suffix
        return os.environ.get(key) or settings.get(key)

    return S3ObjectStore(
        S3Config(
            endpoint=value("ENDPOINT"),
            root_uri=value("ROOT"),
            access_key=value("ACCESS_KEY"),
            secret_key=value("SECRET_KEY"),
            region=value("REGION") or "us-east-1",
        )
    )


def collect(corpus, data, questions, paths, cache, store):
    groups, pair_groups = category_words(questions, data["mapped"])
    benchmark_targets = set().union(*groups.values())
    controls = matched_controls(
        data["tokens"], data["lookup"].counts, benchmark_targets
    )
    targets = sorted(benchmark_targets | set(controls.values()))
    pairs = pair_inventory(pair_groups, data["lookup"].counts)
    shards = json.loads(
        (
            paths.cache_root / "f2/w2v1/corpus-investigation" / f"{corpus}_shards.json"
        ).read_text()
    )
    configuration = dict(
        analysis_version=2,
        corpus=corpus,
        corpus_sha256=data["lookup"].manifest["corpus_digest"],
        vocabulary_digest=data["lookup"].manifest["vocabulary_digest"],
        lexical_token_budget=320_000_000,
        question_sha256=QUESTION_SHA256,
        runs=data["runs"],
        targets=targets,
        benchmark_targets=sorted(benchmark_targets),
        controls={str(k): v for k, v in controls.items()},
        pairs=pairs,
        groups={k: sorted(v) for k, v in groups.items()},
        shards=shards,
        method=dict(
            context_fingerprint="64-bit mixed sum of BLAKE2b token hashes, with bag size; CBOW multiset, not ordered sequence",
            context_sketch_k=256,
            samples_per_word=128,
            context_windows=list(WINDOWS),
            dynamic_weight="(11-distance)/10",
            context_count_storage="uint32; dynamic weights stored as integer tenths",
            exact_sentence_digest="BLAKE2b-128",
            normalized_template="lowercase; digit runs to #; non-alphanumeric to spaces; whitespace collapse",
            bootstrap_seed=20261011,
            bootstrap_repetitions=500,
        ),
    )
    cache.mkdir(parents=True, exist_ok=True)
    config_path = cache / "configuration.json"
    if (cache / "statistics.json").is_file():
        existing = json.loads(config_path.read_text())
        if existing != configuration:
            raise ValueError("Analysis cache configuration differs")
        return existing
    config_path.write_text(json.dumps(configuration, indent=2))
    statistics = CorpusStatistics(
        data["tokens"], data["lookup"].counts, targets, pairs, cache, len(shards)
    )
    statistics.spelling_targets.update(
        spelling_key(t) for q in questions for t in _analogy_tokens(q)
    )
    start = time.monotonic()
    for shard, line in bound_lines(
        shards, store, paths.staging_root / "exp/f2/w2v1/corpus-analysis"
    ):
        statistics.consume(line, shard)
    result = statistics.finish(cache)
    mismatch = np.flatnonzero(statistics.observed != data["lookup"].counts)
    verification = dict(
        corpus_sha256=result["sha256"],
        expected_sha256=configuration["corpus_sha256"],
        sha256_matches=result["sha256"] == configuration["corpus_sha256"],
        lexical_tokens=result["total"]["lexical_tokens"],
        vocabulary_rows=len(statistics.observed),
        vocabulary_count_mismatches=len(mismatch),
        elapsed_seconds=time.monotonic() - start,
        mismatches=[
            dict(
                row=int(r),
                token=data["tokens"][r].decode("utf-8", errors="replace"),
                observed=int(statistics.observed[r]),
                artifact=int(data["lookup"].counts[r]),
            )
            for r in mismatch
        ],
    )
    (cache / "verification.json").write_text(json.dumps(verification, indent=2))
    del statistics
    gc.collect()
    if not verification["sha256_matches"] or len(mismatch):
        raise ValueError(f"Corpus verification failed: {corpus}; see verification.json")
    return configuration


def evaluate(corpus, data, questions, common, paths, cache):
    """Reuse saved LM1B outcomes; cross-corpus runs use the same analogy scorer."""
    path = cache / "question_outcomes.csv"
    if path.is_file():
        return csv_rows(path)
    result = []
    mapped = data["mapped"]
    valid = [i for i, ids in enumerate(mapped) if all(r is not None for r in ids)]
    common_set = set(common)
    prior = (
        paths.analysis_output("f2", "w2v1")
        / "diagnostic/analogy_question_diagnostics.csv"
    )
    if corpus == "lm1b":
        result = csv_rows(prior)
        for row in result:
            row["corpus"] = corpus
            row["common_subset"] = int(row["question_id"]) in common_set
        write_csv(path, result)
        return result
    for run in data["runs"]:
        model, seed, run_id = run["model_kind"], int(run["seed"]), run["mlflow_run_id"]
        print(f"Existing analogy evaluation {corpus} {model} seed {seed}", flush=True)
        lookup = load_lookup_artifact(lookup_path(paths, run_id))
        baseline = json.loads(
            (paths.cache_root / "f2/w2v1/analogy" / f"{run_id}.json").read_text()
        )
        if (
            baseline["signature"]["questions_sha256"] != QUESTION_SHA256
            or baseline["signature"]["vocabulary_limit"] != 82000
        ):
            raise ValueError("Baseline identity mismatch")
        embeddings = lookup.embeddings[:82000]
        outcomes = dict(
            zip(
                valid,
                _analogy_predictions(embeddings, [mapped[i] for i in valid], 256),
                strict=True,
            )
        )
        for section in ("semantic", "syntactic", "total"):
            ids = [
                i
                for i in valid
                if section == "total"
                or questions[i].category.startswith("gram") == (section == "syntactic")
            ]
            accuracy = (
                100
                * sum(
                    bool(outcomes[i][1] and outcomes[i][0] == mapped[i][3]) for i in ids
                )
                / len(ids)
            )
            if abs(accuracy - float(run[f"{section}_accuracy_percent"])) > 1e-10:
                raise ValueError("Aggregate accuracy mismatch")
        for i, q in enumerate(questions):
            included = i in outcomes
            prediction, positive = outcomes.get(i, (-1, False))
            result.append(
                dict(
                    corpus=corpus,
                    model=model,
                    seed=seed,
                    run_id=run_id,
                    question_id=i,
                    category=q.category,
                    word_a=q.a.decode(),
                    word_b=q.b.decode(),
                    word_c=q.c.decode(),
                    word_d=q.expected.decode(),
                    included=included,
                    common_subset=i in common_set,
                    correct=bool(positive and prediction == mapped[i][3])
                    if included
                    else "",
                    prediction=data["tokens"][prediction].decode(errors="replace")
                    if positive
                    else "",
                )
            )
    write_csv(path, result)
    return result


def truth(value):
    return value is True or str(value).lower() == "true"


def prediction_proxy(samples, groups, targets, corpus):
    """Balanced sampled NB center classification, grouped template holdout."""
    index = {r: i for i, r in enumerate(targets)}
    records = []
    for category, words in groups.items():
        words = sorted(words)
        for w, window in enumerate(WINDOWS):
            training = []
            testing = []
            for row in words:
                a, b = [], []
                for _, sentence, four, dynamic in samples[index[row]]:
                    context = four if w == 0 else dynamic
                    if not context:
                        continue
                    (b if hash64(bytes.fromhex(sentence)) % 5 == 0 else a).append(
                        context
                    )
                training.append(a)
                testing.append(b)
            eligible = [i for i, items in enumerate(training) if items and testing[i]]
            if len(eligible) < 2:
                continue
            candidates = [words[i] for i in eligible]
            vocabulary = sorted(
                {r for i in eligible for bag in training[i] + testing[i] for r in bag}
            )
            positions = {r: j for j, r in enumerate(vocabulary)}
            counts = np.ones((len(eligible), len(vocabulary)), dtype=np.float64)
            for i, j in enumerate(eligible):
                for bag in training[j]:
                    for row in bag:
                        counts[i, positions[row]] += 1
            logp = np.log(counts / counts.sum(axis=1, keepdims=True))
            for j, original in enumerate(eligible):
                hits = 0
                for bag in testing[original]:
                    score = logp[:, [positions[r] for r in bag]].sum(axis=1)
                    # Uniform priors; tie contributes its reciprocal rather than label order.
                    tied = np.flatnonzero(
                        np.isclose(score, score.max(), rtol=0, atol=1e-12)
                    )
                    hits += (1 / len(tied)) if j in tied else 0
                records.append(
                    dict(
                        corpus=corpus,
                        category=category,
                        window=window,
                        row=candidates[j],
                        training_examples=len(training[original]),
                        heldout_examples=len(testing[original]),
                        candidate_centers=len(eligible),
                        heldout_accuracy=hits / len(testing[original]),
                        chance_accuracy=1 / len(eligible),
                    )
                )
    return records


def context_features(counts, background):
    p = probability(np.asarray(counts, dtype=np.float64))
    h = entropy(p)
    ratio = np.divide(p, background, out=np.zeros_like(p), where=background > 0)
    return dict(
        context_events=float(np.sum(counts)),
        context_entropy_bits=h,
        effective_context_vocabulary=2**h,
        nonzero_context_words=int(np.count_nonzero(p)),
        top10_context_mass=float(np.sort(p)[-10:].sum()),
        distinctive_context_mass=float(p[ratio >= 4].sum()),
    )


def export_context_counts(contexts, tokens, targets, path):
    """Sparse, lossless Parquet; dense mmap stays a reconstructible cache."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("window", pa.string()),
            ("side", pa.string()),
            ("target_row", pa.int32()),
            ("context_row", pa.int32()),
            ("count_units", pa.uint32()),
            ("unit_scale", pa.int32()),
        ]
    )
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for w, window in enumerate(WINDOWS):
            for side, label in enumerate(("left", "right")):
                for i, row in enumerate(targets):
                    ids = np.flatnonzero(contexts[w, side, i])
                    if not len(ids):
                        continue
                    writer.write_table(
                        pa.Table.from_pydict(
                            dict(
                                window=[window] * len(ids),
                                side=[label] * len(ids),
                                target_row=np.full(len(ids), row, dtype=np.int32),
                                context_row=ids.astype(np.int32),
                                count_units=contexts[w, side, i, ids],
                                unit_scale=np.full(
                                    len(ids), 1 if w == 0 else 10, dtype=np.int32
                                ),
                            ),
                            schema=schema,
                        )
                    )


def summarize_corpus(corpus, data, config, cache, output):
    stats = json.loads((cache / "statistics.json").read_text())
    arrays = np.load(cache / "statistics.npz")
    contexts = np.load(cache / "contexts.npy", mmap_mode="r")
    counts = data["lookup"].counts.astype(np.float64)
    counts[0] = 0
    background = probability(counts)
    targets = config["targets"]
    index = {r: i for i, r in enumerate(targets)}
    tokens = data["tokens"]
    controls = {int(k): v for k, v in config["controls"].items()}
    memberships = defaultdict(list)
    for category, words in config["groups"].items():
        for row in words:
            memberships[row].append(category)
    variants = defaultdict(list)
    for spelling, count in stats["raw_variants"]:
        variants[spelling_key(spelling.encode())].append((spelling, count))
    raw_words = []
    word_rows = []
    top_context = []
    for row in targets:
        i = index[row]
        frequency = float(arrays["word"][i, 0])
        token = tokens[row].decode(errors="replace")
        v = variants[spelling_key(tokens[row])]
        raw_total = sum(c for _, c in v)
        retained_variant = sum(
            c for t, c in v if t.encode() in {tokens[r] for r in targets}
        )
        # All vocabulary spellings count toward retained mass, regardless of target selection.
        retained_variant = sum(
            c for t, c in v if data["lookup"].row(t.encode()) is not None
        )
        shard_counts = arrays["shard_words"][:, i].astype(float)
        active = shard_counts > 0
        shard_probability = probability(shard_counts)
        shard_tokens = np.array(
            [
                stats["shards"].get(str(j), {}).get("lexical_tokens", 0)
                for j in range(len(shard_counts))
            ],
            dtype=float,
        )
        valid = shard_tokens > 0
        rates = np.divide(
            shard_counts[valid],
            shard_tokens[valid],
            out=np.zeros(valid.sum()),
            where=shard_tokens[valid] > 0,
        )
        unique, exact = stats["unique_contexts"][i]
        common = dict(
            corpus=corpus,
            row=row,
            word=token,
            role="benchmark" if row in memberships else "matched_control",
            categories=";".join(memberships[row]),
            frequency=int(frequency),
            vocabulary_frequency=int(data["lookup"].counts[row]),
            raw_spelling_family_frequency=raw_total,
            exact_spelling_share=frequency / raw_total if raw_total else 0,
            spelling_variant_count=len(v),
            spelling_variant_entropy=entropy(
                probability(np.array([c for _, c in v], dtype=float))
            ),
            retained_spelling_family_share=retained_variant / raw_total
            if raw_total
            else 0,
            split_phrase_occurrences=stats["phrase_counts"].get(str(row), 0),
            sentence_occurrences=int(arrays["word"][i, 1]),
            dedup_occurrences=int(arrays["word"][i, 2]),
            normalized_dedup_occurrences=int(arrays["word"][i, 3]),
            dedup_exposure_fraction=arrays["word"][i, 2] / frequency
            if frequency
            else 0,
            normalized_dedup_exposure_fraction=arrays["word"][i, 3] / frequency
            if frequency
            else 0,
            unique_fixed4_contexts=unique,
            unique_context_count_exact=exact,
            context_diversity_per_occurrence=unique / frequency if frequency else 0,
            active_shards=int(active.sum()),
            max_shard_share=float(shard_probability.max()),
            shard_hhi=float(np.sum(shard_probability**2)),
            shard_rate_cv=float(np.std(rates) / np.mean(rates)) if rates.mean() else 0,
            zero_fixed4_context_fraction=arrays["word"][i, 8] / frequency
            if frequency
            else 0,
            fixed4_compressed_distance_fraction=arrays["word"][i, 7]
            / arrays["word"][i, 4]
            if arrays["word"][i, 4]
            else 0,
            repeated_self_context_fraction=arrays["word"][i, 6] / arrays["word"][i, 4]
            if arrays["word"][i, 4]
            else 0,
        )
        for spelling, count in v:
            raw_words.append(
                dict(
                    corpus=corpus,
                    row=row,
                    benchmark_or_control_word=token,
                    raw_spelling=spelling,
                    raw_frequency=count,
                    in_training_vocabulary=data["lookup"].row(spelling.encode())
                    is not None,
                )
            )
        for w, window in enumerate(WINDOWS):
            values = np.asarray(contexts[w, :, i, :].sum(axis=0), dtype=float) / (
                1 if w == 0 else 10
            )
            features = context_features(values, background)
            p = probability(values)
            opposite = js_divergence(
                probability(np.asarray(contexts[w, 0, i], dtype=float)),
                probability(np.asarray(contexts[w, 1, i], dtype=float)),
            )
            word_rows.append(
                dict(
                    common,
                    window=window,
                    **features,
                    context_events_per_occurrence=features["context_events"] / frequency
                    if frequency
                    else 0,
                    raw_distance_context_events=float(arrays["word"][i, 9 + w]),
                    left_right_js_bits=opposite,
                )
            )
            for context in np.argsort(values)[-20:][::-1]:
                top_context.append(
                    dict(
                        corpus=corpus,
                        row=row,
                        word=token,
                        window=window,
                        context_word=tokens[context].decode(errors="replace"),
                        count=float(values[context]),
                        mass=float(p[context]),
                        unigram_lift=float(p[context] / background[context])
                        if background[context]
                        else 0,
                    )
                )
    write_csv(output / "word_features.csv", word_rows)
    write_csv(output / "raw_spelling_variants.csv", raw_words)
    write_csv(output / "top_context_words.csv", top_context)
    write_csv(
        output / "matched_word_controls.csv",
        [
            dict(
                corpus=corpus,
                benchmark_word=tokens[r].decode(errors="replace"),
                control_word=tokens[c].decode(errors="replace"),
                benchmark_frequency=int(counts[r]),
                control_frequency=int(counts[c]),
                log_frequency_difference=float(
                    np.log((counts[c] + 1) / (counts[r] + 1))
                ),
                case_matches=tokens[r][:1].isupper() == tokens[c][:1].isupper(),
                length_difference=len(tokens[c]) - len(tokens[r]),
            )
            for r, c in controls.items()
        ],
    )
    proxy_groups = {k: set(v) for k, v in config["groups"].items()}
    proxy_groups.update(
        {
            k + "-matched-controls": {controls[r] for r in v}
            for k, v in config["groups"].items()
        }
    )
    proxy = prediction_proxy(stats["samples"], proxy_groups, targets, corpus)
    write_csv(output / "conditional_prediction_proxy.csv", proxy)
    write_csv(
        output / "sentence_length_distribution.csv",
        [
            dict(corpus=corpus, sequence=kind, length=int(n), sentences=c)
            for kind, key in (
                ("raw_line", "raw_lengths"),
                ("filtered_training_block", "filtered_lengths"),
            )
            for n, c in stats[key].items()
        ],
    )
    shard_rows = []
    for shard in range(len(config["shards"])):
        if str(shard) not in stats["shards"]:
            continue
        for i, row in enumerate(targets):
            shard_rows.append(
                dict(
                    corpus=corpus,
                    shard=shard,
                    word=tokens[row].decode(errors="replace"),
                    frequency=int(arrays["shard_words"][shard, i]),
                    **stats["shards"][str(shard)],
                )
            )
    write_csv(output / "shard_word_counts.csv", shard_rows)
    if not (cache / "context_counts.parquet").is_file():
        export_context_counts(
            contexts, tokens, targets, cache / "context_counts.parquet"
        )
    write_csv(
        output / "vocabulary_counts.csv",
        [
            dict(
                row=r,
                token=t.decode(errors="replace"),
                observed=int(arrays["observed"][r]),
                artifact=int(data["lookup"].counts[r]),
            )
            for r, t in enumerate(tokens)
        ],
    )
    return word_rows, proxy, stats, arrays, contexts


def relation_features(corpus, data, config, stats, arrays, contexts, word_rows, output):
    tokens = data["tokens"]
    index = {r: i for i, r in enumerate(config["targets"])}
    background = data["lookup"].counts.astype(float)
    background[0] = 0
    background = probability(background)
    patterns = {(p, w): Counter(c) for p, w, c in stats["patterns"]}
    pattern_totals = defaultdict(Counter)
    for i, pair in enumerate(config["pairs"]):
        for w in range(2):
            pattern_totals[pair["category"], pair["kind"], w].update(
                patterns.get((i, w), {})
            )
    word_map = {(r["row"], r["window"]): r for r in word_rows}
    rows = []
    pattern_rows = []
    shard_rows = []
    overlap_rows = []
    for w, window in enumerate(WINDOWS):
        values = np.asarray(contexts[w].sum(axis=0), dtype=float) / (
            1 if w == 0 else 10
        )
        mass = values.sum(axis=1)
        p = np.divide(
            values, mass[:, None], out=np.zeros_like(values), where=mass[:, None] > 0
        )
        for category, words in config["groups"].items():
            ordered = sorted(words, key=lambda r: (int(data["lookup"].counts[r]), r))
            comparisons = set()
            for j, a in enumerate(ordered):
                for b in ordered[max(0, j - 4) : min(len(ordered), j + 5)]:
                    if a != b:
                        comparisons.add(tuple(sorted((a, b))))
            for a, b in sorted(comparisons):
                pa, pb = p[index[a]], p[index[b]]
                overlap_rows.append(
                    dict(
                        corpus=corpus,
                        category=category,
                        window=window,
                        word_a=tokens[a].decode(errors="replace"),
                        word_b=tokens[b].decode(errors="replace"),
                        frequency_a=int(data["lookup"].counts[a]),
                        frequency_b=int(data["lookup"].counts[b]),
                        js_bits=js_divergence(pa, pb),
                        overlap_mass=float(np.minimum(pa, pb).sum()),
                        comparison="same-category-nearest-frequency",
                    )
                )
        for i, pair in enumerate(config["pairs"]):
            a, b = pair["a"], pair["b"]
            ia, ib = index[a], index[b]
            pa, pb = p[ia], p[ib]
            shared = np.minimum(pa, pb)
            general = background > 0
            distinct = (pa >= 4 * background) & (pb >= 4 * background) & general
            direct = float(arrays["shard_pairs"][:, i, w].sum())
            patterns_here = patterns.get((i, w), Counter())
            own = Counter(patterns_here)
            positive = pattern_totals[pair["category"], "relation", w].copy()
            negative = pattern_totals[pair["category"], "negative", w].copy()
            (positive if pair["kind"] == "relation" else negative).subtract(own)
            # Remove the matched control/positive too, to avoid pair-specific leakage.
            matched = pair["matched_relation"]
            for j, other in enumerate(config["pairs"]):
                if j != i and other["matched_relation"] == matched:
                    (positive if other["kind"] == "relation" else negative).subtract(
                        patterns.get((j, w), {})
                    )
            vocabulary = set(positive) | set(negative) | set(own)
            n = max(1, len(vocabulary))
            posmass = sum(positive.values())
            negmass = sum(negative.values())
            pattern_signal = 0.0
            repeat_mass = 0.0
            for pattern, count in own.items():
                loglift = float(
                    np.log2(
                        ((positive[pattern] + 1) / (posmass + n))
                        / ((negative[pattern] + 1) / (negmass + n))
                    )
                )
                pattern_signal += count * loglift
                repeat_mass += count * (positive[pattern] > 0)
                pattern_rows.append(
                    dict(
                        corpus=corpus,
                        pair_id=i,
                        category=pair["category"],
                        kind=pair["kind"],
                        window=window,
                        pattern=pattern,
                        count_units=count,
                        unit_scale=1 if w == 0 else 10,
                        other_positive_count_units=positive[pattern],
                        other_negative_count_units=negative[pattern],
                        leave_pair_out_log2_lift=loglift,
                    )
                )
            pattern_mass = sum(own.values())
            exposure = float(mass[ia])
            freqa = int(data["lookup"].counts[a])
            freqb = int(data["lookup"].counts[b])
            lift = (
                direct / exposure / background[b] if exposure and background[b] else 0
            )
            row = dict(
                corpus=corpus,
                pair_id=i,
                category=pair["category"],
                kind=pair["kind"],
                matched_relation=matched,
                window=window,
                word_a=tokens[a].decode(errors="replace"),
                word_b=tokens[b].decode(errors="replace"),
                frequency_a=freqa,
                frequency_b=freqb,
                min_frequency=min(freqa, freqb),
                direct_cooccurrences=direct,
                direct_per_occurrence=direct / max(1, freqa),
                unigram_log2_lift=float(np.log2(lift)) if lift else -30,
                direct_zero=direct == 0,
                context_overlap=float(shared.sum()),
                context_js_bits=js_divergence(pa, pb),
                shared_distinctive_mass=float(shared[distinct].sum()),
                shared_general_mass=float(
                    shared[
                        background >= np.quantile(background[background > 0], 0.95)
                    ].sum()
                ),
                repeated_relation_pattern_mass=repeat_mass / pattern_mass
                if pattern_mass
                else 0,
                relation_pattern_log2_lift=pattern_signal / pattern_mass
                if pattern_mass
                else 0,
                observed_relation_patterns=len(own),
            )
            for feature in (
                "dedup_exposure_fraction",
                "normalized_dedup_exposure_fraction",
                "context_diversity_per_occurrence",
                "max_shard_share",
                "shard_rate_cv",
                "context_entropy_bits",
                "top10_context_mass",
                "distinctive_context_mass",
                "context_events_per_occurrence",
                "fixed4_compressed_distance_fraction",
                "spelling_variant_entropy",
                "exact_spelling_share",
            ):
                row[feature] = float(
                    np.mean(
                        [word_map[a, window][feature], word_map[b, window][feature]]
                    )
                )
            rows.append(row)
            for shard in range(len(config["shards"])):
                if str(shard) in stats["shards"]:
                    shard_rows.append(
                        dict(
                            corpus=corpus,
                            pair_id=i,
                            category=pair["category"],
                            kind=pair["kind"],
                            window=window,
                            shard=shard,
                            cooccurrences=float(arrays["shard_pairs"][shard, i, w]),
                        )
                    )
        del values, p
    write_csv(output / "relation_features.csv", rows)
    write_csv(output / "relation_patterns.csv", pattern_rows)
    write_csv(output / "shard_relation_counts.csv", shard_rows)
    write_csv(output / "context_overlap_comparisons.csv", overlap_rows)
    return rows, overlap_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--corpus", choices=CORPORA, action="append")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    paths = RuntimePaths.from_environment()
    output = (
        args.output_dir or paths.analysis_output("f2", "w2v1") / "corpus-investigation"
    )
    output.mkdir(parents=True, exist_ok=True)
    raw = (get_benchmark_data_dir(paths) / "questions-words.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != QUESTION_SHA256:
        raise ValueError("Benchmark checksum mismatch")
    questions = parse_analogy_questions(raw.splitlines())
    inputs, common = inventory(paths, questions)
    write_csv(
        output / "common_question_subset.csv",
        [
            dict(
                question_id=i,
                category=questions[i].category,
                **{
                    f"word_{label}": t.decode()
                    for label, t in zip(
                        "abcd", _analogy_tokens(questions[i]), strict=True
                    )
                },
            )
            for i in common
        ],
    )
    store = store_from_environment(paths)
    caches = {}
    for corpus in args.corpus or CORPORA:
        cache = paths.cache_root / "f2/w2v1/corpus-investigation" / corpus
        caches[corpus] = cache
        out = output / corpus
        out.mkdir(parents=True, exist_ok=True)
        print(f"Collecting {corpus}", flush=True)
        config = collect(corpus, inputs[corpus], questions, paths, cache, store)
        if args.collect_only:
            print(f"Verified corpus statistics: {corpus}", flush=True)
            continue
        print(f"Summarizing {corpus}", flush=True)
        word_rows, proxy, stats, arrays, contexts = summarize_corpus(
            corpus, inputs[corpus], config, cache, out
        )
        relation_features(
            corpus, inputs[corpus], config, stats, arrays, contexts, word_rows, out
        )
        evaluate(corpus, inputs[corpus], questions, common, paths, cache)
        del word_rows, proxy, stats, arrays, contexts
        gc.collect()
    if args.collect_only:
        return
    if args.corpus:
        return
    from .corpus_report import build_report

    build_report(inputs, questions, common, caches, output)
    print(f"Report: {output / 'corpus_investigation.md'}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
