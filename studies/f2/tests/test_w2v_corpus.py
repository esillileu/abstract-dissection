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
