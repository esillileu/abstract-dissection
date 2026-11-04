"""Offline readiness and conservative resource bounds for Section 4.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from repro_io.checksum import sha256_file

from .preflight_env import available_memory, resolve_source
from .resource_estimator import (
    disk_free,
    measure_phrase,
    resource_estimate,
)
from .sentences import (
    cached_sentences,
    sentence_cache_path,
    sentences,
)


def _load_statistics(manifest: Path) -> dict | None:
    from f2.suites.common.w2v.formats import VOCABULARY_FORMAT, verify_artifact

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
        vocabulary_path = Path(stored["vocabulary_path"])
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
    from w2v import FileCorpus

    from f2.suites.common.w2v.formats import (
        VOCABULARY_FORMAT,
        verify_artifact,
    )
    from f2.suites.common.w2v.phrases import (
        PhrasePolicy,
        cached_phrase_corpus,
        load_cached_phrase_corpus,
    )
    from f2.suites.common.w2v.preparation import (
        resolve_or_build_vocabulary,
        vocabulary_cache_path,
    )

    source = Path(source)
    cache_root = Path(cache_root)
    staging = staging_root or cache_root
    result = dict(
        source_path=str(source),
        phrase_path=None,
        vocabulary_path=None,
        sentences_cached=False,
        phrases_cached=False,
        vocabulary_cached=False,
        blockers=[],
    )
    if not source.is_file():
        result["blockers"].append(f"source corpus does not exist: {source}")
        result.update(ready=False, status="blocked")
        return result
    source_sha256 = sha256_file(source)
    result["source_sha256"] = source_sha256

    sentence_path = sentence_cache_path(cache_root, source_sha256)
    if not sentence_path.is_file():
        if prepare:
            cached_sentences(source, cache_root, source_sha256)
            result["sentences_cached"] = True
        else:
            result["blockers"].append("sentences not prepared")
    else:
        result["sentences_cached"] = True

    policy = PhrasePolicy.from_config(config["phrases"])
    result["phrase_policy"] = policy.record()
    try:
        phrase_corpus = load_cached_phrase_corpus(cache_root, source_sha256, policy)
        result["phrases_cached"] = True
    except (FileNotFoundError, ValueError):
        if prepare:
            phrase_corpus = cached_phrase_corpus(
                source, cache_root, policy, source_sha256=source_sha256
            )
            result["phrases_cached"] = True
        else:
            phrase_corpus = None
            result["blockers"].append("phrase corpus not prepared")
    if phrase_corpus is not None:
        result["phrase_path"] = str(phrase_corpus.path)
        result["phrase_sha256"] = phrase_corpus.corpus_sha256
        result["phrase_token_count"] = phrase_corpus.token_count
        result["phrase_joined_count"] = phrase_corpus.joined_count

    vocabulary_target = vocabulary_cache_path(
        phrase_corpus.corpus_sha256 if phrase_corpus else source_sha256,
        config["vocabulary"],
        cache_root=cache_root,
        full_tokens=True,
    )
    try:
        verify_artifact(vocabulary_target, format_name=VOCABULARY_FORMAT)
        result["vocabulary_cached"] = True
        result["vocabulary_path"] = str(vocabulary_target)
    except (FileNotFoundError, ValueError):
        if prepare and phrase_corpus is not None:
            corpus = FileCorpus.from_path(phrase_corpus.path, full_tokens=True)
            resolve_or_build_vocabulary(
                corpus,
                config["vocabulary"],
                cache_root=cache_root,
                corpus_digest=phrase_corpus.corpus_sha256,
            )
            result["vocabulary_cached"] = True
            result["vocabulary_path"] = str(vocabulary_target)
        else:
            result["blockers"].append("vocabulary not prepared")

    identity_key = hashlib.sha256(
        json.dumps(
            dict(
                source_sha256=source_sha256,
                phrases=policy.record(),
                vocabulary=config["vocabulary"],
                training=config["training"],
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    stats_manifest = cache_root / "f2/w2v/preflight" / f"{identity_key}.json"
    stats_manifest.parent.mkdir(parents=True, exist_ok=True)
    measured = _load_statistics(stats_manifest)

    if measured is None and phrase_corpus is not None and result["vocabulary_cached"]:
        measured = measure_phrase(
            phrase_corpus.path,
            config["vocabulary"],
            cache_root / "f2/w2v/preflight/scratch",
        )
        stored_record = dict(
            identity_key=identity_key,
            phrase_path=str(phrase_corpus.path),
            phrase_sha256=phrase_corpus.corpus_sha256,
            vocabulary_path=str(vocabulary_target),
            vocabulary_manifest_sha256=sha256_file(vocabulary_target / "manifest.json"),
            **measured,
        )
        payload = json.dumps(stored_record, sort_keys=True)
        stats_manifest.write_text(
            json.dumps(
                dict(
                    statistics_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                    **stored_record,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if measured is not None:
        result["measured"] = {
            key: measured[key]
            for key in (
                "raw_token_count",
                "unique_token_types",
                "retained_token_count",
                "retained_token_types",
                "sentence_profile",
            )
        }
        estimate = resource_estimate(
            config,
            measured_tokens=measured["retained_token_count"],
            measured_retained_types=measured["retained_token_types"],
            source_bytes=source.stat().st_size,
        )
    else:
        estimate = resource_estimate(
            config,
            measured_tokens=int(config["preflight"]["budget_tokens"]),
            measured_retained_types=int(config["preflight"]["budget_vocabulary_types"]),
            source_bytes=source.stat().st_size,
        )
    result["estimate"] = estimate

    ram = available_memory() if available_ram is None else available_ram
    result["available_ram_bytes"] = ram
    if estimate["predicted_peak_rss_bytes"] > ram:
        result["blockers"].append("insufficient physical RAM for peak RSS")
    disk = disk_free(cache_root) if available_disk is None else available_disk
    result["available_disk_bytes"] = disk
    staging_free = disk_free(staging) if available_disk is None else available_disk
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


__all__ = [
    "available_memory",
    "cached_sentences",
    "preflight",
    "resolve_source",
    "resource_estimate",
    "sentence_cache_path",
    "sentences",
]
