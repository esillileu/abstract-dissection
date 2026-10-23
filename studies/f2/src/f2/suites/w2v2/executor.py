"""Word and phrase corpus policies for the shared Word2Vec executor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from f2.suites.w2v.executor import Word2VecExecutor, Word2VecResult, _mapping
from f2.suites.w2v.phrases import PhrasePolicy, cached_phrase_corpus
from repro_core.context import ExperimentContext


class W2V2Executor(Word2VecExecutor):
    """Train word input directly or materialize a phrase corpus first."""

    suite_name = "w2v2"

    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> Word2VecResult:
        mode = config.get("corpus_mode", "phrase")
        if mode == "word":
            if config.get("subsampling_policy") != "word2vec_c":
                raise ValueError("word-only W2V2 requires word2vec_c subsampling")
            if "phrase_detection" in config:
                raise ValueError("word-only W2V2 cannot configure phrase_detection")
            return super().run(config, context)
        if mode != "phrase":
            raise ValueError(f"unsupported W2V2 corpus_mode: {mode}")
        resolved = dict(config)
        corpus = _mapping(config, "corpus")
        source = Path(str(corpus["path"]))
        if not source.is_absolute():
            source = context.paths.repo_root / source
        configured_digest = corpus.get("sha256")
        # Cache preparation hashes actual source bytes and checks any declared identity.
        source_digest = None if configured_digest is None else str(configured_digest)
        values = _mapping(config, "phrase_detection")
        policy = PhrasePolicy.from_config(values)
        progress = context.metadata.get("progress_reporter")
        phrase = cached_phrase_corpus(
            source,
            context.paths.cache_root,
            policy,
            progress=None if progress is None else progress.write,
            source_sha256=source_digest,
        )
        resolved["corpus"] = {"path": str(phrase.path), "sha256": phrase.corpus_sha256}
        result = super().run(resolved, context)
        lineage_target = result.root / "phrase_lineage.json"
        shutil.copyfile(phrase.path.with_suffix(".txt.json"), lineage_target)
        report = json.loads(result.report.read_text())
        report["artifacts"]["phrase_lineage"] = "phrase_lineage.json"
        result.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return result


EXECUTORS = {"word2vec": W2V2Executor()}


def get_executor(kind: str) -> W2V2Executor:
    try:
        return EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown W2V2 experiment kind: {kind}") from exc


__all__ = ["W2V2Executor", "get_executor"]
