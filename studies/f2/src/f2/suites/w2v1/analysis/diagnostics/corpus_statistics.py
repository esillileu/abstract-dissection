"""Streaming corpus exposure, context and relation evidence; no model execution."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

TOKEN_PATTERN = re.compile(rb"[^ \t\r\n]+")

WINDOWS = ("fixed4", "dynamic10_expected")
PUNCTUATION = b".,:;!?()[]{}\"'`"


def spelling_key(token: bytes) -> bytes:
    return token.strip(PUNCTUATION).replace(b"_", b"").replace(b"-", b"").lower()


def training_blocks(fields, vocabulary):
    """Retain original positions while reproducing lexical filtering and boundaries."""
    block = []
    for position, token in enumerate(fields):
        row = vocabulary.get(token.replace(b"\r", b"")[:99].split(b"\0", 1)[0])
        if row is None:
            continue
        if row == 0:
            if block:
                yield block
                block = []
        else:
            block.append((row, position))
            if len(block) == 1000:
                yield block
                block = []
    if block:
        yield block


class BottomHashes:
    """Bounded KMV cardinality estimate; exact when fewer than k signatures occur."""

    def __init__(self, k=256):
        self.k = k
        self.heap = []
        self.values = set()

    def add(self, value):
        if value in self.values:
            return
        if len(self.heap) < self.k:
            self.values.add(value)
            heapq.heappush(self.heap, -value)
        elif value < -self.heap[0]:
            self.values.remove(-heapq.heapreplace(self.heap, -value))
            self.values.add(value)

    def add_many(self, values):
        union = np.unique(np.r_[np.asarray(list(self.values), dtype=np.uint64), values])
        if len(union) > self.k:
            union = np.partition(union, self.k - 1)[: self.k]
        self.values = {int(v) for v in union}
        self.heap = [-v for v in self.values]
        heapq.heapify(self.heap)

    def estimate(self):
        if len(self.heap) < self.k:
            return float(len(self.heap)), True
        return (self.k - 1) * (2**64) / (-self.heap[0] + 1), False


def hash64(value):
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def mix64(value):
    value = np.asarray(value, dtype=np.uint64).copy()
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value


class CorpusStatistics:
    def __init__(self, tokens, counts, targets, pairs, cache: Path, shard_count):
        self.tokens = tokens
        self.counts = np.asarray(counts)
        self.vocabulary = {t: i for i, t in enumerate(tokens)}
        self.targets = targets
        self.target_index = {r: i for i, r in enumerate(targets)}
        self.pairs = pairs
        self.pair_index = defaultdict(list)
        for i, pair in enumerate(pairs):
            self.pair_index[frozenset((pair["a"], pair["b"]))].append(i)
        n = len(targets)
        cache.mkdir(parents=True, exist_ok=True)
        self.contexts = np.lib.format.open_memmap(
            cache / "contexts.npy",
            mode="w+",
            dtype=np.uint32,
            shape=(2, 2, n, len(tokens)),
        )
        self.observed = np.zeros(len(tokens), dtype=np.uint64)
        self.word = np.zeros((n, 11), dtype=np.float64)
        # occurrences, sentences, dedup occurrences, normalized-dedup occurrences,
        # fixed4 slots, dynamic10 expected slots, fixed4 repeated-self slots,
        # fixed4 compressed-distance slots, zero-context occurrences,
        # fixed4 raw-distance slots, dynamic10 raw-distance expected slots.
        self.shard_words = np.zeros((shard_count, n), dtype=np.uint64)
        self.shard_pairs = np.zeros((shard_count, len(pairs), 2), dtype=np.float64)
        self.patterns = defaultdict(Counter)
        self.context_sketches = [BottomHashes() for _ in range(n)]
        self.samples = [[] for _ in range(n)]
        self.sample_seen = [set() for _ in range(n)]
        self.sample_limit = 128
        self.raw_variants = Counter()
        self.spelling_targets = {spelling_key(tokens[r]) for r in targets}
        self.raw_counts = Counter()
        self.exact_sentences = set()
        self.normalized_sentences = set()
        self.raw_lengths = Counter()
        self.filtered_lengths = Counter()
        self.shards = defaultdict(Counter)
        self.digest = hashlib.sha256()
        self.total = Counter()
        self.frequent = set(np.argsort(self.counts[1:])[-500:] + 1)
        self.row_targets = np.full(len(tokens), -1, dtype=np.int32)
        self.row_targets[targets] = np.arange(n)
        self.token_hashes = np.array([hash64(t) for t in tokens], dtype=np.uint64)
        self.pair_keys = np.array(
            sorted(
                {min(p["a"], p["b"]) * len(tokens) + max(p["a"], p["b"]) for p in pairs}
            ),
            dtype=np.int64,
        )
        self.key_pairs = defaultdict(list)
        for i, p in enumerate(pairs):
            self.key_pairs[
                min(p["a"], p["b"]) * len(tokens) + max(p["a"], p["b"])
            ].append(i)
        self.buffer_rows = []
        self.buffer_positions = []
        self.buffer_blocks = []
        self.buffer_relative = []
        self.block_metadata = []
        self.observed_events = []
        self.phrases = {}
        for r in targets:
            token = tokens[r]
            parts = (
                re.sub(rb"([a-z])([A-Z])", rb"\1 \2", token)
                .replace(b"_", b" ")
                .replace(b"-", b" ")
                .lower()
                .split()
            )
            if len(parts) > 1:
                self.phrases.setdefault(parts[0], []).append((tuple(parts), r))
        self.phrase_counts = Counter()

    def flush(self):
        if self.observed_events:
            self.observed += np.bincount(
                self.observed_events, minlength=len(self.tokens)
            ).astype(np.uint64)
            self.observed_events.clear()
        if not self.buffer_rows:
            return
        rows = np.asarray(self.buffer_rows, dtype=np.int64)
        raw = np.asarray(self.buffer_positions, dtype=np.int64)
        blocks = np.asarray(self.buffer_blocks, dtype=np.int64)
        relative = np.asarray(self.buffer_relative, dtype=np.uint64)
        targets = self.row_targets[rows]
        fixed_slots = np.zeros(len(rows), dtype=np.uint64)
        bags = np.zeros(len(rows), dtype=np.uint64)
        n = len(self.targets)
        for distance in range(1, 11):
            for side in (0, 1):
                center = (
                    np.arange(distance, len(rows))
                    if side == 0
                    else np.arange(len(rows) - distance)
                )
                other = center - distance if side == 0 else center + distance
                mask = (blocks[center] == blocks[other]) & (targets[center] >= 0)
                center, other = center[mask], other[mask]
                t = targets[center]
                context = rows[other]
                if not len(t):
                    continue
                np.add.at(self.contexts[1, side], (t, context), 11 - distance)
                self.word[:, 5] += np.bincount(t, minlength=n) * (11 - distance) / 10
                raw_distance = abs(raw[center] - raw[other])
                weights = np.maximum(0, 11 - raw_distance) / 10
                self.word[:, 10] += np.bincount(t, weights=weights, minlength=n)
                self.word[:, 9] += np.bincount(
                    t, weights=raw_distance <= 4, minlength=n
                )
                if distance <= 4:
                    np.add.at(self.contexts[0, side], (t, context), 1)
                    self.word[:, 4] += np.bincount(t, minlength=n)
                    self.word[:, 6] += np.bincount(
                        t, weights=rows[center] == context, minlength=n
                    )
                    self.word[:, 7] += np.bincount(
                        t, weights=raw_distance != distance, minlength=n
                    )
                    fixed_slots[center] += 1
                    bags[center] += self.token_hashes[context]
                if side == 1 and len(self.pair_keys):
                    keys = np.minimum(rows[center], context) * len(
                        self.tokens
                    ) + np.maximum(rows[center], context)
                    places = np.searchsorted(self.pair_keys, keys)
                    matched = (places < len(self.pair_keys)) & (
                        self.pair_keys[np.minimum(places, len(self.pair_keys) - 1)]
                        == keys
                    )
                    for pos, j, key in zip(
                        center[matched], other[matched], keys[matched], strict=True
                    ):
                        shard, _, _ = self.block_metadata[blocks[pos]]
                        for pair_id in self.key_pairs[int(key)]:
                            pair = self.pairs[pair_id]
                            orientation = (
                                "a_left" if pair["a"] == rows[pos] else "a_right"
                            )
                            between = [
                                self.tokens[r].decode("utf-8", errors="replace")
                                if r in self.frequent
                                else "<WORD>"
                                for r in rows[pos + 1 : j]
                            ]
                            pattern = orientation + "|" + " ".join(between)
                            self.patterns[pair_id, 1][pattern] += 11 - distance
                            self.shard_pairs[shard, pair_id, 1] += (11 - distance) / 10
                            if distance <= 4:
                                self.patterns[pair_id, 0][pattern] += 1
                                self.shard_pairs[shard, pair_id, 0] += 1
        valid = np.flatnonzero(targets >= 0)
        self.word[:, 8] += np.bincount(
            targets[valid], weights=fixed_slots[valid] == 0, minlength=n
        )
        fingerprints = mix64(
            bags[valid] ^ (fixed_slots[valid] * np.uint64(0x9E3779B97F4A7C15))
        )
        normcodes = np.array(
            [hash64(bytes.fromhex(meta[1])) for meta in self.block_metadata],
            dtype=np.uint64,
        )
        priorities = mix64(
            normcodes[blocks[valid]]
            ^ self.token_hashes[rows[valid]]
            ^ mix64(relative[valid])
        )
        order = np.argsort(targets[valid], kind="stable")
        sorted_targets = targets[valid[order]]
        bounds = (
            np.r_[0, np.flatnonzero(np.diff(sorted_targets)) + 1, len(order)]
            if len(order)
            else []
        )
        for start, end in pairwise(bounds):
            idx = int(sorted_targets[start])
            group = order[start:end]
            self.context_sketches[idx].add_many(fingerprints[group])
            heap = self.samples[idx]
            limit = -heap[0][0] if len(heap) == self.sample_limit else 2**64
            positions = group[priorities[group] < limit]
            if len(positions) > self.sample_limit:
                positions = positions[
                    np.argpartition(priorities[positions], self.sample_limit - 1)[
                        : self.sample_limit
                    ]
                ]
            for element in positions:
                pos = int(valid[element])
                key = int(priorities[element])
                if key in self.sample_seen[idx] or (
                    len(heap) == self.sample_limit and key >= -heap[0][0]
                ):
                    continue
                _, normsig, rawsig = self.block_metadata[blocks[pos]]
                radius = (
                    1
                    + hash64(
                        bytes.fromhex(rawsig) + int(relative[pos]).to_bytes(4, "little")
                    )
                    % 10
                )

                def context(radius, position=pos):
                    return [
                        int(rows[j])
                        for j in range(
                            max(0, position - radius),
                            min(len(rows), position + radius + 1),
                        )
                        if j != position and blocks[j] == blocks[position]
                    ]

                sample = (-key, normsig, context(4), context(radius))
                if len(heap) == self.sample_limit:
                    old = heapq.heapreplace(heap, sample)
                    self.sample_seen[idx].remove(-old[0])
                else:
                    heapq.heappush(heap, sample)
                self.sample_seen[idx].add(key)
        self.buffer_rows.clear()
        self.buffer_positions.clear()
        self.buffer_blocks.clear()
        self.buffer_relative.clear()
        self.block_metadata.clear()

    def consume(self, line, shard):
        self.digest.update(line)
        original = TOKEN_PATTERN.findall(line)
        fields = (
            TOKEN_PATTERN.findall(line.replace(b"\r", b""))
            if b"\r" in line
            else original
        )
        self.total["lexical_tokens"] += len(original)
        self.total["lines"] += 1
        self.shards[shard]["lexical_tokens"] += len(original)
        self.shards[shard]["lines"] += 1
        self.raw_lengths[len(fields)] += 1
        signature = hashlib.blake2b(line, digest_size=16).digest()
        fresh = signature not in self.exact_sentences
        self.exact_sentences.add(signature)
        normalized = re.sub(rb"\d+", b"#", line.lower())
        normalized = re.sub(rb"[^a-z0-9# ]", b" ", normalized)
        normalized = b" ".join(normalized.split())
        norm_signature = hashlib.blake2b(normalized, digest_size=16).digest()
        norm_fresh = norm_signature not in self.normalized_sentences
        self.normalized_sentences.add(norm_signature)
        self.total["duplicate_lines"] += not fresh
        self.total["normalized_duplicate_lines"] += not norm_fresh
        self.shards[shard]["duplicate_lines"] += not fresh
        self.shards[shard]["normalized_duplicate_lines"] += not norm_fresh
        self.raw_counts.update(fields)
        rows = list(map(self.vocabulary.get, fields))
        if None in rows:
            for position, row in enumerate(rows):
                if row is None:
                    token = fields[position]
                    if len(token) >= 100 or b"\0" in token:
                        rows[position] = self.vocabulary.get(
                            token[:99].split(b"\0", 1)[0]
                        )
        lost = rows.count(None)
        self.total["oov_tokens"] += lost
        self.shards[shard]["oov_tokens"] += lost
        self.observed_events.extend(r for r in rows if r is not None)
        if line.endswith(b"\n"):
            self.observed[0] += 1
        if self.phrases:
            lower = [t.strip(PUNCTUATION).lower() for t in fields]
            for pos, first in enumerate(lower):
                for phrase, row in self.phrases.get(first, ()):
                    if tuple(lower[pos : pos + len(phrase)]) == phrase:
                        self.phrase_counts[row] += 1
        sentence_targets = Counter(r for r in rows if r in self.target_index)
        for row, occurrences in sentence_targets.items():
            i = self.target_index[row]
            self.word[i, :4] += (
                occurrences,
                1,
                occurrences * fresh,
                occurrences * norm_fresh,
            )
            self.shard_words[shard, i] += occurrences
        current = []
        blocks = []
        for position, row in enumerate(rows):
            if row is None:
                continue
            if row == 0:
                if current:
                    blocks.append(current)
                    current = []
                continue
            current.append((row, position))
            if len(current) == 1000:
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)
        for block in blocks:
            self.total["training_blocks"] += 1
            self.filtered_lengths[len(block)] += 1
            self.total["filtered_adjacencies"] += max(0, len(block) - 1)
            self.total["oov_bridged_adjacencies"] += sum(
                block[j][1] - block[j - 1][1] > 1 for j in range(1, len(block))
            )
            block_id = len(self.block_metadata)
            self.block_metadata.append((shard, norm_signature.hex(), signature.hex()))
            self.buffer_rows.extend(row for row, _ in block)
            self.buffer_positions.extend(position for _, position in block)
            self.buffer_blocks.extend([block_id] * len(block))
            self.buffer_relative.extend(range(len(block)))
        if len(self.buffer_rows) >= 500_000:
            self.flush()

    def finish(self, cache):
        self.flush()
        self.contexts.flush()
        np.savez_compressed(
            cache / "statistics.npz",
            observed=self.observed,
            word=self.word,
            shard_words=self.shard_words,
            shard_pairs=self.shard_pairs,
            targets=self.targets,
        )
        for token, count in self.raw_counts.items():
            if len(token) >= 100:
                self.total["truncated_tokens"] += count
            if b"\0" in token:
                self.total["nul_tokens"] += count
            if spelling_key(token) in self.spelling_targets:
                self.raw_variants[token] = count
        records = {
            "sha256": self.digest.hexdigest(),
            "total": dict(self.total),
            "raw_lengths": dict(self.raw_lengths),
            "filtered_lengths": dict(self.filtered_lengths),
            "shards": {str(k): dict(v) for k, v in self.shards.items()},
            "unique_contexts": [s.estimate() for s in self.context_sketches],
            "raw_variants": [
                (t.decode("utf-8", errors="replace"), c)
                for t, c in self.raw_variants.items()
            ],
            "phrase_counts": dict(self.phrase_counts),
            "patterns": [
                (p, w, dict(counts)) for (p, w), counts in self.patterns.items()
            ],
            "samples": [
                [(int(-k), s, a, b) for k, s, a, b in samples]
                for samples in self.samples
            ],
        }
        (cache / "statistics.json").write_text(json.dumps(records), encoding="utf-8")
        return records


def probability(counts):
    mass = float(np.sum(counts))
    return counts / mass if mass else np.zeros_like(counts, dtype=np.float64)


def entropy(p):
    positive = p[p > 0]
    return float(-np.sum(positive * np.log2(positive)))


def js_divergence(p, q):
    mixture = (p + q) / 2
    return entropy(mixture) - (entropy(p) + entropy(q)) / 2


def residualize(values, controls):
    return values - controls @ np.linalg.lstsq(controls, values, rcond=None)[0]


def association(feature, outcome, controls, rng, repetitions=1000, clusters=None):
    """Partial correlation and pairs bootstrap, never question-level inference."""
    x = residualize(feature, controls)
    y = residualize(outcome, controls)

    def corr(a, b):
        return (
            float(np.corrcoef(a, b)[0, 1])
            if np.std(a) > 1e-12 and np.std(b) > 1e-12
            else 0.0
        )

    observed = corr(x, y)
    # Refit nuisance regressions within each pair resample.
    samples = []
    groups = None
    if clusters is not None:
        clusters = np.asarray(clusters)
        groups = [np.flatnonzero(clusters == key) for key in np.unique(clusters)]
    for _ in range(repetitions):
        ids = (
            rng.integers(0, len(x), len(x))
            if groups is None
            else np.concatenate(
                [groups[i] for i in rng.integers(0, len(groups), len(groups))]
            )
        )
        samples.append(
            corr(
                residualize(feature[ids], controls[ids]),
                residualize(outcome[ids], controls[ids]),
            )
        )
    return observed, *np.quantile(samples, (0.025, 0.975))
