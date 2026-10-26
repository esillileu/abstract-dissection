"""Word and phrase corpus policies for the shared Word2Vec executor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from f2.suites.w2v.artifacts import _VOCABULARY_ARRAYS
from f2.suites.w2v.executor import Word2VecExecutor, Word2VecResult, _mapping
from f2.suites.w2v.phrases import PhraseCorpus, PhrasePolicy, cached_phrase_corpus
from repro_core.context import ExperimentContext
from repro_core.storage.local_store import LocalStore


class W2V2Executor(Word2VecExecutor):
    """Train word input directly or materialize a phrase corpus first."""

    suite_name = "w2v2"

    def _run_training(
        self,
        config: dict[str, object],
        context: ExperimentContext,
        *,
        run_root: Path,
        local_store: LocalStore,
        resume_from: Path | None = None,
        progress_reporter: Any | None = None,
    ) -> Word2VecResult:
        mode = config.get("corpus_mode", "phrase")
        if mode == "word":
            if config.get("subsampling_policy") != "word2vec_c":
                raise ValueError("word-only W2V2 requires word2vec_c subsampling")
            if "phrase_detection" in config:
                raise ValueError("word-only W2V2 cannot configure phrase_detection")
            return super()._run_training(
                config,
                context,
                run_root=run_root,
                local_store=local_store,
                resume_from=resume_from,
                progress_reporter=progress_reporter,
            )
        if mode != "phrase":
            raise ValueError(f"unsupported W2V2 corpus_mode: {mode}")

        resolved = dict(config)
        corpus = _mapping(config, "corpus")
        source = Path(str(corpus["path"]))
        if config.get("large_scale"):
            from .preflight import resolve_source

            source = resolve_source(config, context.paths)
        elif not source.is_absolute():
            source = context.paths.repo_root / source

        configured_digest = corpus.get("sha256")
        source_digest = None if configured_digest is None else str(configured_digest)
        readiness = None
        prepared_dir = context.paths.inputs("f2", "prepared")

        if config.get("large_scale"):
            from .preflight import preflight

            readiness = preflight(
                config,
                source,
                prepared_dir,
                staging_root=context.paths.runs_root,
                prepare=True,
            )
            if not readiness["ready"]:
                raise ValueError(
                    "large phrase preflight blocked: "
                    + "; ".join(readiness["blockers"])
                )
            statistics = readiness["statistics"]
            source_digest = statistics["sentence_sha256"]

        values = _mapping(config, "phrase_detection")
        policy = PhrasePolicy.from_config(values)
        progress = context.metadata.get("progress_reporter") or progress_reporter

        if readiness is None:
            phrase = cached_phrase_corpus(
                source,
                prepared_dir,
                policy,
                progress=None if progress is None else progress.write,
                source_sha256=source_digest,
            )
        else:
            phrase_path = Path(readiness["statistics"]["phrase_path"])
            lineage = json.loads(phrase_path.with_suffix(".txt.json").read_text())
            phrase = PhraseCorpus(
                phrase_path,
                lineage["source_sha256"],
                lineage["corpus_sha256"],
                lineage["policy_digest"],
                lineage["lineage_digest"],
                lineage["token_count"],
                lineage["joined_count"],
            )

        resolved["corpus"] = {"path": str(phrase.path), "sha256": phrase.corpus_sha256}
        result = super()._run_training(
            resolved,
            context,
            run_root=run_root,
            local_store=local_store,
            resume_from=resume_from,
            progress_reporter=progress_reporter,
        )

        lineage_target = result.root / "phrase_lineage.json"
        shutil.copyfile(phrase.path.with_suffix(".txt.json"), lineage_target)
        report = json.loads(result.report.read_text(encoding="utf-8"))

        if readiness is not None:
            from repro_io.checksum import sha256_file

            checkpoint_manifest = json.loads(
                (result.checkpoint / "manifest.json").read_text()
            )
            lineage = {
                **config["large_scale"],
                "preflight": readiness,
                "sentence_sha256": source_digest,
                "phrase_sha256": phrase.corpus_sha256,
                "phrase_policy": values,
                "config_digest": config["identity"]["config_digest"],
                "planned_run_slot_id": config["identity"]["planned_run_slot_id"],
                "seed": config["identity"]["seed"],
                "vocabulary_policy": config["vocabulary"],
                "training_policy": config["training"],
                "lookup_manifest_sha256": sha256_file(result.lookup / "manifest.json"),
                "vocabulary_artifact": {
                    "digest": checkpoint_manifest["vocabulary_digest"],
                    "files": {
                        name: checkpoint_manifest["files"][name]
                        for name in _VOCABULARY_ARRAYS
                    },
                },
                "checkpoint_manifest_sha256": sha256_file(
                    result.checkpoint / "manifest.json"
                ),
            }
            (result.root / "large_scale_lineage.json").write_text(
                json.dumps(lineage, indent=2, sort_keys=True) + "\n"
            )
            report["artifacts"]["large_scale_lineage"] = "large_scale_lineage.json"

        report["artifacts"]["phrase_lineage"] = "phrase_lineage.json"
        local_store.write_result_report(report)
        return result


EXECUTORS = {"word2vec": W2V2Executor()}


def get_executor(kind: str) -> W2V2Executor:
    try:
        return EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown W2V2 experiment kind: {kind}") from exc


__all__ = ["W2V2Executor", "get_executor"]
