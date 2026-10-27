from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from f2.suites.w2v.corpus import (
    CorpusBinding,
    CorpusMaterializer,
    CorpusShard,
)
from repro_core.context.paths import RuntimePaths


class FixtureStore:
    def __init__(self, objects: dict[str, Path], *, fail_once: bool = False) -> None:
        self.objects = objects
        self.fail_once = fail_once
        self.calls = 0

    def get_file(self, uri: str, local_path: Path) -> Path:
        self.calls += 1
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if self.fail_once:
            self.fail_once = False
            local_path.write_bytes(b"partial")
            raise OSError("interrupted fixture transfer")
        shutil.copyfile(self.objects[uri], local_path)
        return local_path


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        repo_root=tmp_path,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
        staging_root=tmp_path / "staging",
        references_root=tmp_path / "references",
        studies_root=tmp_path / "studies",
    )


def _fixture_binding(tmp_path: Path) -> tuple[CorpusBinding, dict[str, Path]]:
    contents = [b"one two three\nfour five\n", b"six seven\neight nine ten\n"]
    shards = []
    objects = {}
    for index, content in enumerate(contents):
        raw = tmp_path / f"raw-{index}.txt"
        compressed = tmp_path / f"shard-{index}.txt.zst"
        raw.write_bytes(content)
        subprocess.run(
            ["zstd", "-q", "-f", "-T1", raw.as_posix(), "-o", compressed.as_posix()],
            check=True,
        )
        uri = f"s3://fixture/shard-{index}.txt.zst"
        digest = hashlib.sha256(compressed.read_bytes()).hexdigest()
        shards.append(CorpusShard(index, uri, digest, compressed.stat().st_size, 5, 2))
        objects[uri] = compressed
    return CorpusBinding(tuple(shards)), objects


def test_materializes_in_manifest_order_and_stops_at_exact_token_budget(
    tmp_path: Path,
) -> None:
    binding, objects = _fixture_binding(tmp_path)
    store = FixtureStore(objects)
    progress: list[tuple[int, int, int]] = []
    result = CorpusMaterializer(store, paths=_paths(tmp_path)).materialize(
        binding,
        lexical_token_budget=7,
        progress=lambda *values: progress.append(values),
    )

    assert result.path.read_bytes() == b"one two three\nfour five\nsix seven\n"
    assert result.lexical_tokens == 7
    assert result.complete_shards == 1
    assert progress == [(1, 2, 5), (2, 2, 7)]

    cached = CorpusMaterializer(store, paths=_paths(tmp_path)).materialize(
        binding, lexical_token_budget=7
    )
    assert cached == result
    assert store.calls == 2


def test_binding_is_derived_from_verified_repository_rows() -> None:
    rows = [
        {
            "shard_index": 0,
            "s3_uri": "s3://fixture/shard-00000.txt.zst",
            "sha256": "a" * 64,
            "byte_size": 10,
            "word_count": 3,
            "doc_count": 1,
        }
    ]

    binding = CorpusBinding.from_rows(rows)

    assert binding.shards[0].uri == "s3://fixture/shard-00000.txt.zst"


def test_rejects_reordered_and_corrupt_shards(tmp_path: Path) -> None:
    binding, objects = _fixture_binding(tmp_path)
    reordered = replace(binding, shards=tuple(reversed(binding.shards)))
    with pytest.raises(ValueError, match="ordered with contiguous indices"):
        CorpusMaterializer(FixtureStore(objects), paths=_paths(tmp_path)).materialize(
            reordered, lexical_token_budget=3
        )

    corrupt = replace(binding.shards[0], sha256="0" * 64)
    corrupt_binding = replace(binding, shards=(corrupt, binding.shards[1]))
    with pytest.raises(OSError, match="failed verification"):
        CorpusMaterializer(
            FixtureStore(objects), paths=_paths(tmp_path), download_attempts=1
        ).materialize(corrupt_binding, lexical_token_budget=3)


def test_recovers_from_interrupted_download_and_invalid_cached_result(
    tmp_path: Path,
) -> None:
    binding, objects = _fixture_binding(tmp_path)
    store = FixtureStore(objects, fail_once=True)
    materializer = CorpusMaterializer(store, paths=_paths(tmp_path))
    result = materializer.materialize(binding, lexical_token_budget=10)
    assert (
        result.path.read_bytes()
        == b"one two three\nfour five\nsix seven\neight nine ten\n"
    )
    assert store.calls == 3

    result.path.write_bytes(b"truncated")
    rebuilt = materializer.materialize(binding, lexical_token_budget=10)
    assert rebuilt.lexical_tokens == 10
    assert rebuilt.path.read_bytes().endswith(b"eight nine ten\n")
    assert store.calls == 3


def test_manifest_cache_skips_stream_and_rebuilds_on_size_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, objects = _fixture_binding(tmp_path)
    store = FixtureStore(objects)
    materializer = CorpusMaterializer(store, paths=_paths(tmp_path))

    # First call: materializes and writes manifest
    first = materializer.materialize(binding, lexical_token_budget=7)

    # Second call: monkeypatch _write_corpus to fail if called
    def fail_if_written(*args, **kwargs):
        raise AssertionError("unexpected _write_corpus call on manifest cache hit")

    monkeypatch.setattr(materializer, "_write_corpus", fail_if_written)
    second = materializer.materialize(binding, lexical_token_budget=7)
    assert second == first

    # Corrupt the file size -> manifest cache hit should fail and rebuild
    first.path.write_bytes(b"short")
    rebuilt_calls = 0
    orig_write = CorpusMaterializer._write_corpus

    def counted_write(*args, **kwargs):
        nonlocal rebuilt_calls
        rebuilt_calls += 1
        return orig_write(materializer, *args, **kwargs)

    monkeypatch.setattr(materializer, "_write_corpus", counted_write)
    third = materializer.materialize(binding, lexical_token_budget=7)
    assert third == first
    assert rebuilt_calls == 1


def test_recovers_same_size_corruption_of_materialized_corpus(tmp_path):
    binding, objects = _fixture_binding(tmp_path)
    materializer = CorpusMaterializer(FixtureStore(objects), paths=_paths(tmp_path))
    original = materializer.materialize(binding, lexical_token_budget=10)
    data = original.path.read_bytes()
    original.path.write_bytes(b"x" * len(data))
    rebuilt = materializer.materialize(binding, lexical_token_budget=10)
    assert rebuilt.path.read_bytes() == data
    assert rebuilt.corpus_sha256 == original.corpus_sha256


def test_concurrent_materialization_uses_private_work_and_verified_results(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    binding, objects = _fixture_binding(tmp_path)
    materializer = CorpusMaterializer(FixtureStore(objects), paths=_paths(tmp_path))
    budgets = (3, 7, 10, 3, 7, 10)
    with ThreadPoolExecutor(3) as pool:
        results = list(
            pool.map(
                lambda budget: materializer.materialize(
                    binding, lexical_token_budget=budget
                ),
                budgets,
            )
        )
    for i in range(3):
        assert results[i] == results[i + 3]
        assert len(results[i].path.read_bytes().split()) == budgets[i]
    assert not list((materializer._corpus_root() / "work").iterdir())


def test_shards_without_final_newline_cannot_join_documents(tmp_path):
    source = []
    for i, data in enumerate((b"alpha", b"beta\n")):
        raw = tmp_path / f"raw-{i}"
        raw.write_bytes(data)
        compressed = tmp_path / f"shard-{i}.zst"
        subprocess.run(
            ["zstd", "-q", "-f", str(raw), "-o", str(compressed)], check=True
        )
        source.append(compressed)
    binding = CorpusBinding(
        tuple(
            CorpusShard(
                i,
                f"s3://fixture/{i}",
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_size,
                1,
                1,
            )
            for i, path in enumerate(source)
        )
    )
    store = FixtureStore({f"s3://fixture/{i}": path for i, path in enumerate(source)})
    result = CorpusMaterializer(store, paths=_paths(tmp_path)).materialize(
        binding, lexical_token_budget=2
    )
    assert result.path.read_bytes() == b"alpha\nbeta\n"


def test_swapped_manifest_does_not_supply_another_budget_corpus(tmp_path):
    from f2.suites.w2v.corpus import _binding_manifest_key

    materializer_binding, objects = _fixture_binding(tmp_path)
    materializer = CorpusMaterializer(FixtureStore(objects), paths=_paths(tmp_path))
    small = materializer.materialize(materializer_binding, lexical_token_budget=3)
    large = materializer.materialize(materializer_binding, lexical_token_budget=10)
    manifests = materializer._corpus_root() / "manifests"
    small_manifest = (
        manifests / f"{_binding_manifest_key(materializer_binding, 3)}.json"
    )
    large_manifest = (
        manifests / f"{_binding_manifest_key(materializer_binding, 10)}.json"
    )
    large_manifest.write_bytes(small_manifest.read_bytes())
    recovered = materializer.materialize(materializer_binding, lexical_token_budget=10)
    assert recovered == large and recovered != small


def test_slice_corpus_file(tmp_path: Path):
    from f2.suites.w2v.corpus import slice_corpus_file

    source = tmp_path / "raw.txt"
    source.write_bytes(b"word1 word2 word3\nword4 word5 word6\nword7 word8\n")

    target = tmp_path / "sliced-4.txt"
    tokens, sha256 = slice_corpus_file(source, target, 4)
    assert tokens == 4
    assert target.is_file()
    assert target.read_bytes() == b"word1 word2 word3\nword4\n"

    # Verify cache hit
    tokens_cached, sha256_cached = slice_corpus_file(source, target, 4)
    assert tokens_cached == 4
    assert sha256_cached == sha256

    # Verify cache hit when budget exceeds available tokens in source
    target_over = tmp_path / "sliced-over.txt"
    tokens_over, sha256_over = slice_corpus_file(source, target_over, 50)
    assert tokens_over == 8
    tokens_over_cached, sha256_over_cached = slice_corpus_file(source, target_over, 50)
    assert tokens_over_cached == 8
    assert sha256_over_cached == sha256_over
