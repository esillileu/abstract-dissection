"""Offline readiness and conservative resource bounds for Section 4.1."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import psutil
from repro_io.checksum import sha256_file

from f2.suites.w2v.phrases import _tokens

SENTENCE_POLICY = "terminal-punctuation-or-lf-v1"
MAX_SENTENCE_TOKENS = 65_536
MAX_TOKEN_BYTES = 4096


def resolve_source(config: dict, paths, override: Path | None = None) -> Path:
    """Inject the configured dataset root; absolute overrides are explicit inputs."""
    candidate = override or Path(str(config["corpus"]["path"]))
    return (
        candidate
        if candidate.is_absolute()
        else paths.dataset("f2") / "corpus" / candidate
    )


def available_memory(
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
) -> int:
    """Use the smallest remaining host/current-cgroup/ancestor budget."""
    available = psutil.virtual_memory().available
    relative = next(
        (
            line.split("::", 1)[1]
            for line in proc_cgroup.read_text().splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if relative is None:
        raise ValueError("cgroup v2 memory accounting is required")
    # A namespaced mount can expose the current cgroup directly at the root.
    cgroup_root = cgroup_root.resolve()
    leaf = (cgroup_root / relative.lstrip("/")).resolve()
    if not leaf.is_relative_to(cgroup_root) or not leaf.is_dir():
        leaf = cgroup_root
    while True:
        maximum, current = leaf / "memory.max", leaf / "memory.current"
        if maximum.is_file() and current.is_file():
            value = maximum.read_text().strip()
            if value != "max":
                available = min(
                    available, max(0, int(value) - int(current.read_text()))
                )
        if leaf == cgroup_root:
            break
        leaf = leaf.parent
    return available


def sentences(source: Path):
    """Reconstruct boundaries after .!? (optional closing quotes), and at LF.

    LF remains a hard boundary: no cross-document joins. Abbreviations may be
    over-segmented; this is an explicit reconstruction, not a linguistic claim.
    """
    sentence = []
    for token in _tokens(source, max_token_bytes=MAX_TOKEN_BYTES):
        if token is not None:
            sentence.append(token)
            if len(sentence) > MAX_SENTENCE_TOKENS:
                raise ValueError(
                    "sentence exceeds 65536 raw tokens; splitting is forbidden"
                )
        if token is None or token.rstrip(b"\"'\xe2\x80\x9d)]}").endswith(
            (b".", b"!", b"?")
        ):
            if sentence:
                yield sentence
                sentence = []
    if sentence:
        yield sentence


def _disk_free(path: Path) -> int:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _profile(lengths: dict[int, int]) -> dict:
    count = sum(lengths.values())
    tokens = sum(n * frequency for n, frequency in lengths.items())
    pairs = sum(n * (n - 1) * frequency for n, frequency in lengths.items())

    def percentile(fraction):
        position = max(1, math.ceil(count * fraction))
        seen = 0
        for length, frequency in sorted(lengths.items()):
            seen += frequency
            if seen >= position:
                return length
        return 0

    largest = sorted(lengths.items(), reverse=True)
    tail_count = max(1, math.ceil(count * 0.01))
    tail_pairs = 0
    for n, frequency in largest:
        take = min(frequency, tail_count)
        tail_pairs += take * n * (n - 1)
        tail_count -= take
        if tail_count == 0:
            break
    return dict(
        sentences=count,
        lexical_tokens=tokens,
        max_sentence_tokens=max(lengths, default=0),
        p50=percentile(0.5),
        p95=percentile(0.95),
        p99=percentile(0.99),
        ordered_pairs=pairs,
        longest_one_percent_pair_fraction=tail_pairs / pairs if pairs else 0,
        length_histogram={
            str(n): frequency for n, frequency in sorted(lengths.items())
        },
    )


def _measure_phrase(path: Path, vocabulary_config: dict, work: Path) -> dict:
    """Count exact types with bounded SQLite memory before allocating a vocabulary."""
    work.mkdir(parents=True, exist_ok=True)
    for abandoned in work.glob("vocabulary-counts-*"):
        shutil.rmtree(abandoned)
    with tempfile.TemporaryDirectory(
        prefix="vocabulary-counts-", dir=work
    ) as temporary:
        with sqlite3.connect(Path(temporary) / "counts.sqlite") as db:
            db.execute("PRAGMA cache_size=-16384")
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA temp_store=FILE")
            db.execute(
                "CREATE TABLE words(token BLOB PRIMARY KEY, n INTEGER, first_seen INTEGER) WITHOUT ROWID"
            )
            db.execute("CREATE INDEX ranked ON words(n DESC,first_seen)")
            batch = []
            histogram = Counter()
            length = total = line_bytes = max_line_bytes = 0

            def flush():
                db.executemany(
                    "INSERT INTO words VALUES (?,1,?) ON CONFLICT(token) DO UPDATE SET n=n+1",
                    batch,
                )
                db.commit()
                batch.clear()

            for token in _tokens(path, max_token_bytes=16_387):
                if token is None:
                    if length:
                        histogram[length] += 1
                    max_line_bytes = max(max_line_bytes, line_bytes)
                    length = line_bytes = 0
                else:
                    length += 1
                    line_bytes += len(token) + 1
                    total += 1
                    if length > MAX_SENTENCE_TOKENS:
                        raise ValueError("phrase sentence exceeds hard limit")
                    batch.append((token, total))
                    if len(batch) >= 4096:
                        flush()
            if length:
                histogram[length] += 1
            flush()
            unique, byte_count = db.execute(
                "SELECT count(*),coalesce(sum(length(token)),0) FROM words"
            ).fetchone()
            limit = int(vocabulary_config.get("max_lexical_words", 0)) or -1
            selected = db.execute(
                "SELECT count(*),coalesce(sum(length(token)),0),coalesce(sum(n),0) FROM (SELECT token,n FROM words WHERE n>=? ORDER BY n DESC,first_seen LIMIT ?)",
                (int(vocabulary_config.get("min_count", 5)), limit),
            ).fetchone()
            return dict(
                unique_words=unique,
                all_token_bytes=byte_count,
                selected_words=selected[0] + 1,
                selected_token_bytes=selected[1] + 4,
                selected_occurrences=selected[2],
                phrase_bytes=path.stat().st_size,
                max_sentence_bytes=max(max_line_bytes, line_bytes),
                sentence_distribution=_profile(histogram),
            )


def resource_estimate(config: dict, statistics: dict) -> dict:
    """Model/IO costs from actual prepared bytes, vocabulary and sentence histogram."""
    training = config["training"]
    words = int(statistics["vocabulary"]["words"])
    dimension = int(training["embedding_dimension"])
    matrices = 8 * words * dimension
    vocabulary = int(statistics["vocabulary"]["memory_bytes"])
    entry_memory = vocabulary - int(statistics["vocabulary"]["hash_memory_bytes"])
    workers = int(training["thread_count"])
    profile = statistics["retained_sentence_distribution"]
    worker = workers * (
        profile["max_sentence_tokens"] * 16 + dimension * 12 + 64 * 1024
    )
    interval = int(training.get("observation_interval", 0))
    observations = (
        (statistics["selected_occurrences"] // interval + workers) * 1024
        if interval
        else 0
    )
    sigmoid = int(training["sigmoid_table_size"]) * 4
    history_bytes = (
        (
            (statistics["selected_occurrences"] // interval + workers)
            * int(training["epochs"])
            * 1024
        )
        if interval
        else 0
    )
    history_bytes += int(training["epochs"]) * 512
    checkpoint = (
        matrices
        + statistics["vocabulary_artifact_bytes"]
        + workers * 512
        + history_bytes
        + 65536
    )
    # Previous verified + new generation + at most one downloaded matrix during verification.
    verification_file = max(
        matrices // 2 + 128,
        statistics["vocabulary_max_file_bytes"],
        history_bytes,
        65536,
    )
    disk = checkpoint * 2 + verification_file + history_bytes
    training_ram = (
        matrices
        + vocabulary
        + sigmoid
        + worker
        + observations
        + int(training["epochs"]) * 512
        + 32 * 1024**2
    )
    vocabulary_load_ram = (
        vocabulary
        + entry_memory * 2
        + statistics["vocabulary_artifact_bytes"] * 2
        + 32 * 1024**2
    )
    ram = max(training_ram, vocabulary_load_ram)
    epochs = int(training["epochs"])
    pairs = profile["ordered_pairs"] * epochs
    depth = statistics["vocabulary"]["max_huffman_depth"]
    return dict(
        vocabulary_words=words,
        model_bytes=matrices,
        sigmoid_bytes=sigmoid,
        vocabulary_bytes=vocabulary,
        checkpoint_bytes=checkpoint,
        ram_bytes=ram,
        additional_staging_disk_bytes=disk,
        ordered_pairs_upper_bound=pairs,
        hs_node_updates_upper_bound=pairs * depth,
        coordinate_updates_upper_bound=pairs * depth * dimension,
        corpus_read_bytes=statistics["phrase_bytes"] * epochs,
        shard_alignment_read_bytes_upper_bound=2
        * max(0, workers - 1)
        * statistics["max_sentence_bytes"],
        assumptions="actual phrase/vocabulary; pair bound before OOV/subsampling; chunked matrix IO; latest+new+remote verification; checkpoint lookup view",
    )


def _load_statistics(manifest: Path) -> dict | None:
    from f2.suites.w2v.artifacts import VOCABULARY_FORMAT, verify_artifact

    try:
        stored = json.loads(manifest.read_text())
        checksum = stored.pop("statistics_sha256")
        if stored.get("identity_key") != manifest.stem:
            return None
        if (
            checksum
            != hashlib.sha256(json.dumps(stored, sort_keys=True).encode()).hexdigest()
        ):
            return None
        if sha256_file(Path(stored["phrase_path"])) != stored["phrase_sha256"]:
            return None
        vocabulary_path = Path(stored["vocabulary_path"])
        if (
            sha256_file(vocabulary_path / "manifest.json")
            != stored["vocabulary_manifest_sha256"]
        ):
            return None
        verify_artifact(vocabulary_path, format_name=VOCABULARY_FORMAT)
        return stored
    except (ValueError, KeyError, OSError, TypeError):
        return None


def preflight(
    config: dict,
    source: Path,
    cache_root: Path,
    *,
    available_ram: int | None = None,
    available_disk: int | None = None,
    staging_root: Path | None = None,
    prepare: bool = False,
) -> dict:
    """Inspect cached real statistics; preparation is explicit and never trains."""
    from w2v import Corpus, TrainingConfig, VocabularyConfig

    from f2.suites.w2v.artifacts import (
        resolve_or_build_vocabulary,
        vocabulary_cache_path,
    )
    from f2.suites.w2v.phrases import (
        PhrasePolicy,
        cached_phrase_corpus,
        load_cached_phrase_corpus,
    )

    result = dict(
        artifact_id=config["large_scale"]["artifact_id"],
        paper_lexical_tokens=config["large_scale"]["paper_lexical_tokens"],
        expected_lexical_tokens=config["large_scale"]["lexical_tokens"],
        lexical_shortfall=config["large_scale"]["paper_lexical_tokens"]
        - config["large_scale"]["lexical_tokens"],
        source_path=str(source),
        sentence_policy=SENTENCE_POLICY,
        execution_policy="contiguous-sentence-shards-v2",
        ready=False,
        status="blocked",
        blockers=[],
    )
    try:
        TrainingConfig(**config["training"])
        VocabularyConfig(**config["vocabulary"])
    except (TypeError, ValueError, OverflowError) as exc:
        result["blockers"].append(f"invalid engine policy: {exc}")
        return result
    if not source.is_file():
        result["blockers"].append(
            "local verified source unavailable; no download attempted"
        )
        return result
    ram = available_memory() if available_ram is None else available_ram
    disk = _disk_free(cache_root) if available_disk is None else available_disk
    result.update(available_ram_bytes=ram, available_disk_bytes=disk)
    # These allocations depend on configured reservations, not the selected vocabulary size.
    reserve = int(config["vocabulary"].get("initial_capacity", 1000)) * 80
    hash_bytes = int(config["vocabulary"].get("hash_capacity", 30_000_000)) * 16
    sigmoid = int(config["training"]["sigmoid_table_size"]) * 4
    result["configured_reservation_bytes"] = reserve + hash_bytes + sigmoid
    if reserve + hash_bytes + sigmoid + 32 * 1024**2 > ram:
        result["blockers"].append(
            "insufficient RAM for initial vocabulary/hash/sigmoid reservations"
        )
        return result
    parsing_bound = (
        min(source.stat().st_size, MAX_SENTENCE_TOKENS * MAX_TOKEN_BYTES) * 2
        + min(source.stat().st_size, MAX_SENTENCE_TOKENS) * 64
    )
    if parsing_bound + 32 * 1024**2 > ram:
        result["blockers"].append(
            "insufficient RAM for bounded source sentence parsing"
        )
        return result
    digest = sha256_file(source)
    result["source_sha256"] = digest
    if config["corpus"].get("sha256", digest) != digest:
        result["blockers"].append("source SHA256 mismatch")
        return result
    policy = PhrasePolicy.from_config(config["phrase_detection"])
    key = hashlib.sha256(
        json.dumps(
            dict(
                format="actual-vocabulary-statistics",
                source=digest,
                sentence=SENTENCE_POLICY,
                phrase=policy.record(),
                vocabulary=config["vocabulary"],
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    root = cache_root / "f2/w2v/preflight"
    manifest = root / f"{key}.json"
    statistics = _load_statistics(manifest)
    if statistics is None:
        histogram = Counter()
        max_sentence_bytes = max_token_bytes = 0
        try:
            for sentence in sentences(source):
                histogram[len(sentence)] += 1
                max_sentence_bytes = max(
                    max_sentence_bytes, sum(map(len, sentence)) + len(sentence)
                )
                max_token_bytes = max(
                    max_token_bytes, max(map(len, sentence), default=0)
                )
        except ValueError as exc:
            result["blockers"].append(str(exc))
            return result
        raw = {
            **_profile(histogram),
            "max_sentence_bytes": max_sentence_bytes,
            "max_token_bytes": max_token_bytes,
        }
        result.update(
            lexical_tokens=raw["lexical_tokens"], source_sentence_distribution=raw
        )
        if raw["lexical_tokens"] != int(config["large_scale"]["lexical_tokens"]):
            result["blockers"].append(
                f"expected exactly {config['large_scale']['lexical_tokens']} lexical tokens before phrases; found {raw['lexical_tokens']}"
            )
        preparation_ram = (
            max(
                reserve + hash_bytes,
                raw["max_sentence_bytes"] * 2 + raw["max_sentence_tokens"] * 64,
            )
            + 32 * 1024**2
        )
        preparation_disk = source.stat().st_size * 8 + raw["lexical_tokens"] * 256
        result["preparation_estimate"] = dict(
            ram_bytes=preparation_ram, additional_cache_disk_bytes=preparation_disk
        )
        if preparation_ram > ram:
            result["blockers"].append(
                "insufficient RAM for sentence/vocabulary preparation"
            )
        if preparation_disk > disk:
            result["blockers"].append(
                "insufficient local disk for phrase/count preparation"
            )
        if result["blockers"]:
            return result
        if not prepare:
            result.update(status="requires_preparation")
            result["blockers"].append(
                "actual phrase vocabulary statistics unavailable; run explicit --prepare"
            )
            return result
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"source-{digest}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if _load_statistics(manifest) is not None:
                return preflight(
                    config,
                    source,
                    cache_root,
                    available_ram=available_ram,
                    available_disk=available_disk,
                    staging_root=staging_root,
                )
            segmented = sentence_cache_path(cache_root, digest)
            try:
                receipt = json.loads(segmented.with_suffix(".json").read_text())
                if (
                    receipt["source_sha256"] != digest
                    or receipt["policy"] != SENTENCE_POLICY
                ):
                    raise ValueError("sentence receipt identity mismatch")
                sentence_sha = receipt["sha256"]
                phrase = load_cached_phrase_corpus(cache_root, sentence_sha, policy)
            except (OSError, ValueError, KeyError, TypeError):
                segmented = cached_sentences(source, cache_root, digest)
                sentence_sha = sha256_file(segmented)
                phrase = cached_phrase_corpus(
                    segmented, cache_root, policy, source_sha256=sentence_sha
                )
            # Only the tiny provenance receipt is retained; the shared phrase artifact is canonical.
            segmented.unlink(missing_ok=True)
            measured = _measure_phrase(
                phrase.path, config["vocabulary"], root / f"counts-{digest}"
            )
            if (
                measured["unique_words"] + 1
                > int(config["vocabulary"]["hash_capacity"]) * 7 // 10
            ):
                result["blockers"].append(
                    "hash capacity too small for exact vocabulary; capacity pruning would change counts"
                )
                return result
            build_ram = (
                hash_bytes
                + max(reserve, (measured["unique_words"] + 1) * 80)
                + (measured["unique_words"] + 1) * 1200
                + measured["all_token_bytes"] * 4
            )
            if build_ram + sigmoid + 32 * 1024**2 > ram:
                result["blockers"].append(
                    "insufficient RAM for measured vocabulary construction"
                )
                return result
            corpus = Corpus(phrase.path, full_tokens=True)
            vocabulary = resolve_or_build_vocabulary(
                corpus,
                config["vocabulary"],
                cache_root=cache_root,
                corpus_digest=phrase.corpus_sha256,
            )
            vocab_path = vocabulary_cache_path(
                phrase.corpus_sha256,
                config["vocabulary"],
                cache_root=cache_root,
                full_tokens=True,
            )
            statistics = {
                "identity_key": key,
                **measured,
                "source_lexical_tokens": raw["lexical_tokens"],
                "source_sentence_distribution": raw,
                "sentence_sha256": sentence_sha,
                "phrase_path": str(phrase.path),
                "phrase_sha256": phrase.corpus_sha256,
                "vocabulary_path": str(vocab_path),
                "vocabulary": vocabulary.statistics(),
                "retained_sentence_distribution": _profile(
                    vocabulary.sentence_lengths(corpus)
                ),
                "vocabulary_manifest_sha256": sha256_file(vocab_path / "manifest.json"),
                "vocabulary_max_file_bytes": max(
                    p.stat().st_size for p in vocab_path.iterdir() if p.is_file()
                ),
                "vocabulary_artifact_bytes": sum(
                    p.stat().st_size for p in vocab_path.iterdir() if p.is_file()
                ),
            }
            temporary = manifest.with_suffix(".tmp")
            statistics_checksum = hashlib.sha256(
                json.dumps(statistics, sort_keys=True).encode()
            ).hexdigest()
            temporary.write_text(
                json.dumps(
                    {**statistics, "statistics_sha256": statistics_checksum},
                    sort_keys=True,
                )
                + "\n"
            )
            temporary.replace(manifest)
            # This was a reconstructible preparation intermediate; the shared final phrase corpus remains.
            segmented.unlink(missing_ok=True)

    if statistics["source_lexical_tokens"] != int(
        config["large_scale"]["lexical_tokens"]
    ):
        result["blockers"].append("prepared source lexical budget mismatch")
    if statistics["vocabulary"]["words"] <= 1:
        result["blockers"].append(
            "no lexical vocabulary survives the configured minimum count"
        )
    estimate = resource_estimate(config, statistics)
    result.update(
        statistics=statistics,
        estimate=estimate,
        lexical_tokens=statistics["source_lexical_tokens"],
    )
    if estimate["ram_bytes"] > ram:
        result["blockers"].append(
            "insufficient RAM for measured training/checkpoint peak"
        )
    staging_free = (
        _disk_free(staging_root or cache_root)
        if available_disk is None
        else available_disk
    )
    result["available_staging_disk_bytes"] = staging_free
    if estimate["additional_staging_disk_bytes"] > staging_free:
        result["blockers"].append(
            "insufficient staging disk for checkpoints/remote verification"
        )
    result.update(
        ready=not result["blockers"],
        status="ready" if not result["blockers"] else "blocked",
    )
    return result


def sentence_cache_path(cache_root: Path, source_sha256: str) -> Path:
    key = hashlib.sha256(
        f"{source_sha256}:{SENTENCE_POLICY}:{MAX_SENTENCE_TOKENS}:{MAX_TOKEN_BYTES}".encode()
    ).hexdigest()
    return cache_root / "f2/w2v/sentences" / f"{key}.txt"


def cached_sentences(source: Path, cache_root: Path, source_sha256: str) -> Path:
    if sha256_file(source) != source_sha256:
        raise ValueError("source SHA256 mismatch before sentence preparation")
    root = cache_root / "f2/w2v/sentences"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{source_sha256}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _cached_sentences(source, cache_root, source_sha256)


def _cached_sentences(source: Path, cache_root: Path, source_sha256: str) -> Path:
    """Atomically cache reconstructed sentences by bytes and policy."""
    target = sentence_cache_path(cache_root, source_sha256)
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    manifest = target.with_suffix(".json")
    if target.is_file() and manifest.is_file():
        try:
            data = json.loads(manifest.read_text())
            if (
                data.get("source_sha256") == source_sha256
                and data.get("policy") == SENTENCE_POLICY
                and data.get("sha256") == sha256_file(target)
            ):
                return target
        except (ValueError, OSError):
            pass
    for abandoned in root.glob(f".{target.stem}-*"):
        abandoned.unlink()
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}-", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for sentence in sentences(source):
                output.write(b" ".join(sentence) + b"\n")
        if sha256_file(source) != source_sha256:
            raise ValueError("source changed during sentence preparation")
        os.replace(name, target)
        manifest.write_text(
            json.dumps(
                dict(
                    source_sha256=source_sha256,
                    sha256=sha256_file(target),
                    policy=SENTENCE_POLICY,
                ),
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        Path(name).unlink(missing_ok=True)
    return target
