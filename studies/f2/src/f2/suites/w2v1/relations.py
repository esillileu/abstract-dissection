"""Read-only corpus context and final-embedding relation diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

import numpy as np
from repro_io.checksum import sha256_file
from repro_io.s3 import S3Config, S3ObjectStore

from f2.common.paths import get_benchmark_data_dir
from f2.suites.w2v.artifacts import LookupEmbeddings, load_lookup_artifact
from f2.suites.w2v.corpus import _decompressed_chunks
from f2.suites.w2v.evaluation import (
    _analogy_rows,
    _analogy_tokens,
    parse_analogy_questions,
)
from repro_core.context import RuntimePaths

from .analysis import QUESTIONS_WORDS_SHA256
from .corpus_statistics import TOKEN_PATTERN
from .diagnostic import MODELS, SEEDS, TABLE3_RUNS


def corpus_lines(path):
    """Reassemble documents for the line-based W2V1 context diagnostics."""
    pending = bytearray()
    with closing(_decompressed_chunks(path)) as chunks:
        for chunk in chunks:
            pieces = chunk.split(b"\n")
            for piece in pieces[:-1]:
                pending.extend(piece)
                pending.append(10)
                yield bytes(pending)
                pending.clear()
            pending.extend(pieces[-1])
        if pending:
            yield bytes(pending)


CATEGORIES = ("capital-world", "city-in-state", "family")
WINDOWS = ("fixed4", "fixed10", "dynamic10_expected")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)


def relation_statistics(vectors: np.ndarray, pairs: list[tuple[int, int]]):
    """Equal-weight unique directed pairs, with self/shared-endpoint controls."""
    differences = vectors[[b for _, b in pairs]] - vectors[[a for a, _ in pairs]]
    norms = np.linalg.norm(differences, axis=1)
    nonzero = norms > 0
    directions = unit_rows(differences)
    valid = directions[nonzero]
    gram = valid @ valid.T
    pair_indices = np.flatnonzero(nonzero)
    upper = np.triu(np.ones(gram.shape, dtype=bool), 1)
    disjoint = upper.copy()
    for i, original_i in enumerate(pair_indices):
        for j, original_j in enumerate(pair_indices):
            if set(pairs[original_i]) & set(pairs[original_j]):
                disjoint[i, j] = False
    resultant = float(np.linalg.norm(valid.mean(axis=0))) if len(valid) else None
    total = valid.sum(axis=0)
    loo = np.full(len(pairs), np.nan)
    if len(valid) > 1:
        loo[nonzero] = np.sum(valid * unit_rows(total[None, :] - valid), axis=1)
    stats = {
        "unique_pairs": len(pairs),
        "zero_relation_vectors": int((~nonzero).sum()),
        "mean_direction_length": resultant,
        "mean_pairwise_cosine": float(gram[upper].mean()) if upper.any() else None,
        "median_pairwise_cosine": float(np.median(gram[upper]))
        if upper.any()
        else None,
        "mean_disjoint_pairwise_cosine": float(gram[disjoint].mean())
        if disjoint.any()
        else None,
        "disjoint_comparisons": int(disjoint.sum()),
        "mean_relation_norm": float(norms.mean()),
    }
    return stats, norms, loo


def question_relations(
    vectors: np.ndarray, rows: list[tuple[int, int, int, int]]
) -> np.ndarray:
    ids = np.asarray(rows)
    left = unit_rows(vectors[ids[:, 1]] - vectors[ids[:, 0]])
    right = unit_rows(vectors[ids[:, 3]] - vectors[ids[:, 2]])
    return np.sum(left * right, axis=1)


def category_inputs(lookup: LookupEmbeddings, question_bytes: bytes):
    if hashlib.sha256(question_bytes).hexdigest() != QUESTIONS_WORDS_SHA256:
        raise ValueError("Question checksum mismatch")
    questions = parse_analogy_questions(question_bytes.splitlines())
    wanted = {token.upper() for q in questions for token in _analogy_tokens(q)}
    rows = _analogy_rows(lookup, 82000, wanted=wanted)  # Recorded evaluation boundary.
    groups = {}
    for category in CATEGORIES:
        pairs = set()
        valid = []
        total = 0
        for q in questions:
            if q.category != category:
                continue
            total += 1
            ids = [rows.get(t.upper()) for t in _analogy_tokens(q)]
            if any(r is None for r in ids):
                continue
            valid.append(tuple(ids))
            pairs.update(((ids[0], ids[1]), (ids[2], ids[3])))
        groups[category] = {
            "pairs": sorted(pairs),
            "questions": valid,
            "total_questions": total,
        }
    return groups


def decode_tokens(lookup: LookupEmbeddings) -> list[bytes]:
    return [
        bytes(lookup.token_bytes[a:b])
        for a, b in zip(
            lookup.token_offsets[:-1], lookup.token_offsets[1:], strict=True
        )
    ]


def word_inventory(lookup, questions, groups, tokens):
    inventory = []
    wanted = {
        token.upper()
        for q in parse_analogy_questions(questions.splitlines())
        for token in _analogy_tokens(q)
    }
    evaluation_rows = _analogy_rows(lookup, 82000, wanted=wanted)
    for category in CATEGORIES:
        words = sorted(
            {
                token.upper()
                for q in parse_analogy_questions(questions.splitlines())
                if q.category == category
                for token in _analogy_tokens(q)
            }
        )
        analyzed = {row for pair in groups[category]["pairs"] for row in pair}
        for word in words:
            row = evaluation_rows.get(word)
            inventory.append(
                dict(
                    category=category,
                    benchmark_word=word.decode("ascii"),
                    included_in_evaluation=row is not None,
                    used_in_relation_diagnostics=row in analyzed
                    if row is not None
                    else False,
                    training_token=tokens[row].decode("ascii")
                    if row is not None
                    else "",
                    training_frequency=int(lookup.counts[row])
                    if row is not None
                    else "",
                )
            )
    return inventory


def geometry_diagnostics(cache_root: Path, output: Path, paths: RuntimePaths):
    rows = []
    pair_rows = []
    question_rows = []
    sample = None
    questions = (get_benchmark_data_dir(paths) / "questions-words.txt").read_bytes()
    for model in MODELS:
        for seed in SEEDS:
            run_id = TABLE3_RUNS[model][seed]
            print(f"Geometry {model} seed {seed}", flush=True)
            lookup = load_lookup_artifact(cache_root / run_id / "lookup")
            baseline = json.loads(
                (paths.cache_root / "f2/w2v1/analogy" / f"{run_id}.json").read_text()
            )
            if lookup.manifest != baseline["lookup_identity"]:
                raise ValueError(f"Lookup identity mismatch: {run_id}")
            if sample is None:
                sample = lookup
            for field in ("counts", "token_bytes", "token_offsets"):
                if not np.array_equal(getattr(sample, field), getattr(lookup, field)):
                    raise ValueError(f"Vocabulary differs: {run_id}")
            if lookup.manifest["corpus_digest"] != sample.manifest["corpus_digest"]:
                raise ValueError(f"Corpus differs: {run_id}")
            groups = category_inputs(lookup, questions)
            tokens = decode_tokens(lookup)
            # Diagnose raw v(b)-v(a) and the normalized operands used by the evaluator.
            for representation in ("raw_embedding", "unit_embedding"):
                vectors = np.asarray(lookup.embeddings, dtype=np.float64)
                if representation == "unit_embedding":
                    vectors = unit_rows(vectors)
                for category, group in groups.items():
                    pairs = group["pairs"]
                    stats, norms, loo = relation_statistics(vectors, pairs)
                    scores = question_relations(vectors, group["questions"])
                    identity = dict(
                        model=model,
                        seed=seed,
                        run_id=run_id,
                        representation=representation,
                        category=category,
                    )
                    rows.append(
                        dict(
                            identity,
                            **stats,
                            included_questions=len(scores),
                            total_questions=group["total_questions"],
                            question_relation_cosine_mean=float(scores.mean()),
                            question_relation_cosine_median=float(np.median(scores)),
                        )
                    )
                    for (a, b), norm, alignment in zip(pairs, norms, loo, strict=True):
                        pair_rows.append(
                            dict(
                                identity,
                                word_a=tokens[a].decode("ascii"),
                                word_b=tokens[b].decode("ascii"),
                                frequency_a=int(lookup.counts[a]),
                                frequency_b=int(lookup.counts[b]),
                                relation_norm=float(norm),
                                leave_one_out_alignment=float(alignment),
                            )
                        )
                    for qid, (ids, score) in enumerate(
                        zip(group["questions"], scores, strict=True)
                    ):
                        question_rows.append(
                            dict(
                                identity,
                                category_question_index=qid,
                                **{
                                    f"word_{suffix}": tokens[idx].decode("ascii")
                                    for suffix, idx in zip("abcd", ids, strict=True)
                                },
                                relation_cosine=float(score),
                            )
                        )
    write_csv(
        output / "category_word_inventory.csv",
        word_inventory(sample, questions, groups, tokens),
    )
    write_csv(output / "relation_geometry_by_seed.csv", rows)
    write_csv(output / "relation_pairs.csv", pair_rows)
    write_csv(output / "relation_question_cosines.csv", question_rows)
    summary = []
    measures = (
        "mean_direction_length",
        "mean_pairwise_cosine",
        "mean_disjoint_pairwise_cosine",
        "question_relation_cosine_mean",
    )
    for model in MODELS:
        for representation in ("raw_embedding", "unit_embedding"):
            for category in CATEGORIES:
                selected = [
                    r
                    for r in rows
                    if r["model"] == model
                    and r["representation"] == representation
                    and r["category"] == category
                ]
                record = dict(
                    model=model,
                    representation=representation,
                    category=category,
                    seeds=len(selected),
                    unique_pairs=selected[0]["unique_pairs"],
                    included_questions=selected[0]["included_questions"],
                )
                for field in measures:
                    values = [r[field] for r in selected]
                    record[field + "_mean"] = float(np.mean(values))
                    record[field + "_sd"] = float(np.std(values, ddof=1))
                summary.append(record)
    write_csv(output / "relation_geometry_summary.csv", summary)
    return sample, groups, tokens


def bound_corpus_lines(
    shards: list[dict], store: S3ObjectStore, staging: Path, budget: int
):
    """Match the materializer bytes, buffering only one compressed shard on disk."""
    tokens = 0
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="context-", dir=staging) as tmp:
        source = Path(tmp) / "shard.txt.zst"
        for shard in shards:
            print(
                f"Corpus shard {shard['shard_index']}; lexical tokens {tokens:,}/{budget:,}",
                flush=True,
            )
            store.get_file(shard["s3_uri"], source)
            if (
                source.stat().st_size != int(shard["byte_size"])
                or sha256_file(source) != shard["sha256"]
            ):
                raise ValueError("Corpus shard checksum mismatch")
            for line in corpus_lines(source):
                matches = list(TOKEN_PATTERN.finditer(line))
                available = budget - tokens
                if len(matches) <= available:
                    yield line
                    tokens += len(matches)
                else:
                    if available:
                        yield (
                            line[: matches[available - 1].end()].rstrip(b"\r\n") + b"\n"
                        )
                        tokens += available
                    break
            source.unlink()
            if tokens == budget:
                return
    if tokens != budget:
        raise ValueError(f"Insufficient corpus: {tokens} != {budget}")


def context_counts(lines: Iterable[bytes], tokens: list[bytes], targets: list[int]):
    """Sentence-local windows after vocabulary filtering, no subsampling."""
    exact_rows = {t: i for i, t in enumerate(tokens)}
    target_indices = {row: i for i, row in enumerate(targets)}
    matrices = np.zeros((3, len(targets), len(tokens)), dtype=np.uint64)
    observed = np.zeros(len(targets), dtype=np.uint64)
    digest = hashlib.sha256()
    lexical = 0
    sentences = 0
    for line in lines:
        digest.update(line)
        fields = TOKEN_PATTERN.findall(line)
        lexical += len(fields)
        # The engine removes CR, truncates to 99 bytes, and uses exact case.
        ids = [
            exact_rows.get(t.replace(b"\r", b"")[:99].split(b"\0", 1)[0])
            for t in TOKEN_PATTERN.findall(line.replace(b"\r", b""))
        ]
        # OOV tokens are skipped before window distances are measured.
        ids = [r for r in ids if r is not None]
        blocks = []
        sentence = []
        for row in ids:
            if row == 0:
                if sentence:
                    blocks.append(sentence)
                    sentence = []
                continue
            sentence.append(row)
            if len(sentence) == 1000:
                blocks.append(sentence)
                sentence = []
        if sentence:
            blocks.append(sentence)
        for sentence in blocks:
            sentences += 1
            for position, row in enumerate(sentence):
                target = target_indices.get(row)
                if target is None:
                    continue
                observed[target] += 1
                for j in range(
                    max(0, position - 10), min(len(sentence), position + 11)
                ):
                    distance = abs(position - j)
                    if not distance:
                        continue
                    context = sentence[j]
                    if distance <= 4:
                        matrices[0, target, context] += 1
                    matrices[1, target, context] += 1
                    # Uniform shrinking radius 1..10: P(radius >= distance).
                    matrices[2, target, context] += 11 - distance
    return (
        matrices,
        observed,
        dict(
            corpus_sha256=digest.hexdigest(),
            lexical_tokens=lexical,
            sentence_blocks=sentences,
        ),
    )


def context_diagnostics(lines, lookup, groups, tokens, output):
    targets = sorted(
        {r for group in groups.values() for pair in group["pairs"] for r in pair}
    )
    target_indices = {row: i for i, row in enumerate(targets)}
    matrices, observed, provenance = context_counts(lines, tokens, targets)
    if provenance["corpus_sha256"] != lookup.manifest["corpus_digest"]:
        raise ValueError("Streamed corpus SHA-256 differs from training corpus")
    if not np.array_equal(observed, lookup.counts[targets]):
        raise ValueError(
            "Target occurrence counts differ from saved training vocabulary"
        )
    provenance.update(target_words=len(targets), verified_target_counts=True)
    pair_rows = []
    summary = []
    word_rows = []
    background = lookup.counts.astype(np.float64)
    background[0] = 0
    background /= background.sum()
    for window, counts in zip(WINDOWS, matrices, strict=True):
        values = counts.astype(np.float64)
        if window == "dynamic10_expected":
            values /= 10
        mass = values.sum(axis=1)
        probability = np.divide(
            values, mass[:, None], out=np.zeros_like(values), where=mass[:, None] != 0
        )
        ratio = np.divide(
            probability,
            background[None, :],
            out=np.zeros_like(values),
            where=background[None, :] != 0,
        )
        ppmi = np.zeros_like(values)
        np.log2(ratio, out=ppmi, where=ratio > 0)
        np.maximum(ppmi, 0, out=ppmi)
        profiles = unit_rows(ppmi)
        for i, row in enumerate(targets):
            word_rows.append(
                dict(
                    window=window,
                    word=tokens[row].decode("ascii"),
                    frequency=int(observed[i]),
                    context_events=float(mass[i]),
                    events_per_occurrence=float(mass[i] / observed[i]),
                    nonzero_contexts=int((values[i] > 0).sum()),
                )
            )
        for category, group in groups.items():
            pairs = group["pairs"]
            mapped_pairs = [(target_indices[a], target_indices[b]) for a, b in pairs]
            stats, _, _ = relation_statistics(profiles, mapped_pairs)
            category_pairs = []
            for a, b in pairs:
                ia, ib = target_indices[a], target_indices[b]
                direct = float(values[ia, b])
                association = (
                    float(np.log2(probability[ia, b] / background[b]))
                    if direct
                    else None
                )
                row = dict(
                    category=category,
                    window=window,
                    word_a=tokens[a].decode("ascii"),
                    word_b=tokens[b].decode("ascii"),
                    frequency_a=int(observed[ia]),
                    frequency_b=int(observed[ib]),
                    direct_cooccurrences=direct,
                    direct_per_million_a=float(direct / observed[ia] * 1e6),
                    unigram_log2_lift=association,
                    context_ppmi_cosine=float(profiles[ia] @ profiles[ib]),
                )
                pair_rows.append(row)
                category_pairs.append(row)
            summary.append(
                dict(
                    category=category,
                    window=window,
                    unique_pairs=len(pairs),
                    zero_direct_pairs=sum(
                        r["direct_cooccurrences"] == 0 for r in category_pairs
                    ),
                    direct_cooccurrences_sum=sum(
                        r["direct_cooccurrences"] for r in category_pairs
                    ),
                    direct_cooccurrences_median=float(
                        np.median([r["direct_cooccurrences"] for r in category_pairs])
                    ),
                    direct_per_million_a_median=float(
                        np.median([r["direct_per_million_a"] for r in category_pairs])
                    ),
                    context_ppmi_cosine_mean=float(
                        np.mean([r["context_ppmi_cosine"] for r in category_pairs])
                    ),
                    context_relation_pairwise_cosine=stats["mean_pairwise_cosine"],
                    context_relation_disjoint_cosine=stats[
                        "mean_disjoint_pairwise_cosine"
                    ],
                )
            )
    write_csv(output / "context_pair_signals.csv", pair_rows)
    write_csv(output / "context_signal_summary.csv", summary)
    write_csv(output / "context_word_signals.csv", word_rows)
    (output / "context_provenance.md").write_text(
        "# Corpus verification\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in provenance.items())
        + "\n"
    )
    return summary


def report(output: Path, context_summary: list[dict] | None):
    with (output / "relation_geometry_summary.csv").open() as stream:
        geometry = list(csv.DictReader(stream))
    lines = [
        "# Table 3 context and relation geometry",
        "",
        "## Relation-vector consistency",
        "",
        "Pairs are extracted from the same fully included questions as the preceding analogy diagnosis. OOV questions are excluded; category_word_inventory.csv records the full benchmark word inventory. Each unique directed pair has equal weight; repeated analogy appearances are deduplicated. Each seed is analyzed separately, then seed means and sample SDs are reported. Larger cosine means more consistent directions.",
        "",
        "Raw: v(b)-v(a). Unit: normalize word vectors before subtraction (the evaluator operands). All relation vectors are unit-normalized before comparison; diagnostics use float64 arithmetic on saved float32 embeddings. Disjoint comparisons exclude any shared word between pairs, controlling repeated country/state targets.",
        "",
        "| Category | Operands | CBOW pair cosine | SG pair cosine | CBOW disjoint cosine | SG disjoint cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for category in CATEGORIES:
        for representation in ("raw_embedding", "unit_embedding"):
            selected = {
                r["model"]: r
                for r in geometry
                if r["category"] == category and r["representation"] == representation
            }
            values = []
            for field in ("mean_pairwise_cosine", "mean_disjoint_pairwise_cosine"):
                for model in MODELS:
                    r = selected[model]
                    values.append(
                        f"{float(r[field + '_mean']):.4f} ± {float(r[field + '_sd']):.4f}"
                    )
            lines.append(
                f"| {category} | {representation} | " + " | ".join(values) + " |"
            )
    lines.extend(
        [
            "",
            "These metrics do not perform vocabulary ranking, so an observed difference is present in the saved representations independently of nearest-neighbor evaluation. They do not identify the training cause. Shared endpoints, category composition, vector norms and embedding anisotropy can affect consistency; both operand definitions and disjoint comparisons are reported.",
            "",
            "## Corpus windows",
            "",
        ]
    )
    if context_summary is None:
        lines.append(
            "Corpus context analysis has not been run; no corpus conclusions are inferred from embedding geometry."
        )
    else:
        for category in CATEGORIES:
            four = next(
                r
                for r in context_summary
                if r["category"] == category and r["window"] == "fixed4"
            )
            ten = next(
                r
                for r in context_summary
                if r["category"] == category and r["window"] == "fixed10"
            )
            lines.append(
                f"- {category}: {four['unique_pairs'] - four['zero_direct_pairs']}/{four['unique_pairs']} pairs already cooccur in window 4; window 10 adds {four['zero_direct_pairs'] - ten['zero_direct_pairs']} previously unseen pairs. Median direct count {four['direct_cooccurrences_median']:.1f} -> {ten['direct_cooccurrences_median']:.1f}; PPMI context similarity {four['context_ppmi_cosine_mean']:.4f} -> {ten['context_ppmi_cosine_mean']:.4f}; context relation disjoint cosine {four['context_relation_disjoint_cosine']:.4f} -> {ten['context_relation_disjoint_cosine']:.4f}."
            )
        lines.append("")
        fields = list(context_summary[0])
        lines.extend(
            [
                "| " + " | ".join(fields) + " |",
                "| " + " | ".join("---" for _ in fields) + " |",
            ]
        )
        for r in context_summary:
            lines.append(
                "| "
                + " | ".join(
                    f"{r[f]:.4f}" if isinstance(r[f], float) else str(r[f])
                    for f in fields
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Windows are radii, not total widths. Fixed4 and fixed10 count every context within ±4/±10 retained lexical positions; dynamic10_expected weights distance d by (11-d)/10. Actual per-seed shrinking RNG and thread byte-start boundaries are not replayed. No subsampling was used in these runs. Newlines and 1,000 retained-token blocks terminate contexts; OOV tokens are removed before measuring distance. Corpus SHA-256 and all analyzed word counts match training.",
                "",
                "Context profiles are unigram-reference PPMI: max(0, log2(P(context|target)/P_unigram(context))). These are descriptive corpus statistics, not learned CBOW states. Context averaging, gradients and hierarchical-softmax updates are not simulated. Direct cooccurrence growth alone is insufficient evidence of stronger relation signal: larger windows add opportunities. Report zero-pair counts, frequency-normalized rates, per-pair unigram lift, PPMI context similarity and relation directions in normalized PPMI profiles separately.",
            ]
        )
    lines.extend(
        [
            "",
            "No retraining, hyperparameter change or training implementation change was performed.",
            "",
        ]
    )
    observations = ["## Observed comparisons", ""]
    for category in CATEGORIES:
        selected = {
            r["model"]: r
            for r in geometry
            if r["category"] == category and r["representation"] == "raw_embedding"
        }
        cbow = float(selected["cbow"]["mean_disjoint_pairwise_cosine_mean"])
        sg = float(selected["skipgram"]["mean_disjoint_pairwise_cosine_mean"])
        observations.append(
            f"- {category}: raw relation disjoint cosine CBOW {cbow:.4f}, SG {sg:.4f}; SG - CBOW = {sg - cbow:+.4f}. This is a representation comparison independent of vocabulary ranking."
        )
    observations.extend(
        [
            "- The family control is included in the same geometry comparison; geography-specific degradation must be assessed relative to this control. Absolute alignment is not an accuracy measure comparable across categories.",
            "",
        ]
    )
    if context_summary is not None:
        for category in CATEGORIES:
            four = next(
                r
                for r in context_summary
                if r["category"] == category and r["window"] == "fixed4"
            )
            ten = next(
                r
                for r in context_summary
                if r["category"] == category and r["window"] == "fixed10"
            )
            dynamic = next(
                r
                for r in context_summary
                if r["category"] == category and r["window"] == "dynamic10_expected"
            )
            observations.append(
                f"- {category}: direct exposure sum fixed4 {four['direct_cooccurrences_sum']:.1f}, fixed10 {ten['direct_cooccurrences_sum']:.1f} ({ten['direct_cooccurrences_sum'] / four['direct_cooccurrences_sum']:.3f}x), dynamic10 expectation {dynamic['direct_cooccurrences_sum']:.1f} ({dynamic['direct_cooccurrences_sum'] / four['direct_cooccurrences_sum']:.3f}x). These are exposure counts, not gradient magnitudes."
            )
        observations.extend(
            [
                "- Direct exposure growth, PPMI context similarity and context relation alignment measure different properties. Read the zero-pair counts and relation cosines alongside exposure growth before inferring a window effect.",
                "- Corpus/context transformations, CBOW-specific training and optimization remain candidates. This analysis does not isolate window size from architecture or establish an implementation defect.",
                "",
            ]
        )
    lines[2:2] = observations
    (output / "relation_context_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--bound-shards", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.corpus_path and args.bound_shards:
        parser.error("Choose corpus-path or bound-shards")
    paths = RuntimePaths.from_environment()
    output = (
        args.output_dir
        or paths.analysis_output("f2", "w2v1") / "diagnostic" / "relations"
    )
    output.mkdir(parents=True, exist_ok=True)
    roots = [
        r
        for r in (paths.cache_root / "mlflow_artifact").iterdir()
        if all(
            (r / TABLE3_RUNS[m][s] / "lookup/manifest.json").is_file()
            for m in MODELS
            for s in SEEDS
        )
    ]
    if args.cache_root is None and len(roots) != 1:
        raise ValueError("Specify cache-root containing the six run lookups")
    lookup, groups, tokens = geometry_diagnostics(
        args.cache_root or roots[0], output, paths
    )
    summary = None
    report(output, summary)
    if args.corpus_path:
        with args.corpus_path.open("rb") as stream:
            summary = context_diagnostics(stream, lookup, groups, tokens, output)
    elif args.bound_shards:
        from dotenv import dotenv_values

        settings = dotenv_values(paths.repo_root / ".env")

        def value(suffix):
            key = "F2_CORPUS_S3_" + suffix
            return os.environ.get(key) or settings.get(key)

        store = S3ObjectStore(
            S3Config(
                endpoint=value("ENDPOINT"),
                root_uri=value("ROOT"),
                access_key=value("ACCESS_KEY"),
                secret_key=value("SECRET_KEY"),
                region=value("REGION") or "us-east-1",
            )
        )
        shards = json.loads(args.bound_shards.read_text())
        lines = bound_corpus_lines(
            shards, store, paths.staging_root / "exp/f2/w2v1/context", 320_000_000
        )
        summary = context_diagnostics(lines, lookup, groups, tokens, output)
    report(output, summary)
    print(f"Report: {output / 'relation_context_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
