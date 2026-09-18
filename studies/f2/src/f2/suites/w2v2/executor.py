"""Phrase policy layered on the shared W2V epoch executor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from repro_io.checksum import sha256_file

from f2.suites.w2v.phrases import PhrasePolicy, materialize_phrase_corpus
from f2.suites.w2v1.executor import W2V1Executor, W2V1Result, _mapping
from repro_core.context import ExperimentContext


class W2V2Executor(W2V1Executor):
    suite_name = "w2v2"

    def run(self, config: dict[str, object], context: ExperimentContext) -> W2V1Result:
        resolved = dict(config)
        corpus = _mapping(config, "corpus")
        source = Path(str(corpus["path"]))
        if not source.is_absolute():
            source = context.paths.repo_root / source
        configured_digest = corpus.get("sha256")
        # Canonical materialization has already verified configured digests;
        # unbound local inputs are hashed here for phrase lineage.
        source_digest = (
            str(configured_digest)
            if configured_digest is not None
            else sha256_file(source)
        )
        values = _mapping(config, "phrase_detection")
        policy = PhrasePolicy(
            passes=int(values["passes"]),
            threshold=float(values["threshold"]),
            min_count=int(values["min_count"]),
            separator=str(values.get("separator", "_")).encode("ascii"),
        )
        slot = str(_mapping(config, "identity")["planned_run_slot_id"])
        phrase_path = (
            context.paths.run_staging(
                domain="f2",
                suite="w2v2",
                study="phrase-corpus",
                variant="derived",
                run_key=slot,
            )
            / "phrases.txt"
        )
        progress = context.metadata.get("progress_reporter")
        phrase = materialize_phrase_corpus(
            source,
            phrase_path,
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
