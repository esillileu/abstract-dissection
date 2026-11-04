"""Word and phrase corpus policies for W2V2 execution."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from f2.suites.common.w2v.adapter import (
    execute_w2v_training,
    run_w2v_with_coordinator,
)
from f2.suites.common.w2v.phrases import (
    PhraseCorpus,
    PhrasePolicy,
    cached_phrase_corpus,
)
from f2.suites.common.w2v.training import (
    W2VRunReceipt,
    Word2VecResult,
    mapping,
)
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.storage.local_store import LocalStore


class W2V2Executor:
    """Train word input directly or materialize a phrase corpus first."""

    suite_name = "w2v2"

    def execute_spec(
        self,
        config: dict[str, object],
        *,
        tracking_uri: str,
        paths: RuntimePaths,
        override: bool = False,
        progress_reporter: Any | None = None,
    ) -> W2VRunReceipt:
        return run_w2v_with_coordinator(
            config,
            suite_name=self.suite_name,
            tracking_uri=tracking_uri,
            paths=paths,
            override=override,
            progress_reporter=progress_reporter,
            training_step=self._run_training_step,
        )

    def _run_training_step(
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
            return execute_w2v_training(
                config,
                context,
                suite_name=self.suite_name,
                run_root=run_root,
                local_store=local_store,
                resume_from=resume_from,
                progress_reporter=progress_reporter,
            )
        if mode != "phrase":
            raise ValueError(f"unsupported W2V2 corpus_mode: {mode}")

        resolved = dict(config)
        corpus = mapping(config, "corpus")
        source = Path(str(corpus["path"]))

        policy = PhrasePolicy.from_config(mapping(config, "phrase_detection"))
        progress = context.metadata.get("progress_reporter")
        configured_digest = corpus.get("sha256")
        phrase = cached_phrase_corpus(
            source,
            context.paths.inputs("f2", "prepared"),
            policy,
            progress=None if progress is None else progress.write,
            source_sha256=str(configured_digest) if configured_digest else None,
        )
        self._record_phrase_lineage(run_root, phrase)
        resolved["corpus"] = {
            "path": str(phrase.path),
            "sha256": phrase.corpus_sha256,
            "lexical_tokens": phrase.token_count,
        }
        return execute_w2v_training(
            resolved,
            context,
            suite_name=self.suite_name,
            run_root=run_root,
            local_store=local_store,
            resume_from=resume_from,
            progress_reporter=progress_reporter,
        )

    @staticmethod
    def _record_phrase_lineage(run_root: Path, phrase: PhraseCorpus) -> None:
        lineage_src = phrase.path.with_suffix(".txt.json")
        if lineage_src.is_file():
            shutil.copyfile(lineage_src, run_root / "phrase_lineage.json")
            metadata_dir = run_root / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(lineage_src, metadata_dir / "phrase_lineage.json")


EXECUTORS = {"word2vec": W2V2Executor()}


def get_executor(kind: str = "word2vec") -> W2V2Executor:
    try:
        return EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown W2V2 experiment kind: {kind}") from exc


__all__ = [
    "EXECUTORS",
    "W2V2Executor",
    "get_executor",
]
