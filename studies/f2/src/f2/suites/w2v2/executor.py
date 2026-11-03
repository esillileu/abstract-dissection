"""Word and phrase corpus policies for W2V2 execution."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.storage.local_store import LocalStore

from f2.suites.common.w2v.adapter import (
    execute_w2v_training,
    prepare_benchmarks,
    run_w2v_with_coordinator,
)
from f2.suites.common.w2v.artifacts import _VOCABULARY_ARRAYS
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
        if config.get("large_scale"):
            from .preflight import resolve_source

            source = resolve_source(config, context.paths)
        elif not source.is_absolute():
            source = context.paths.repo_root / source

        phrase = cached_phrase_corpus(
            source,
            PhrasePolicy.from_config(mapping(config, "phrase_detection")),
            cache_root=context.paths.inputs("f2", "prepared"),
        )
        self._record_phrase_lineage(run_root, phrase)
        resolved["corpus"] = {
            "path": str(phrase.path),
            "sha256": phrase.sha256,
            "lexical_tokens": phrase.lexical_tokens,
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
        metadata_dir = run_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "phrase_lineage.json").write_text(
            json.dumps(
                {
                    "source_corpus": str(phrase.source_path),
                    "source_sha256": phrase.source_sha256,
                    "target_corpus": str(phrase.path),
                    "target_sha256": phrase.sha256,
                    "passes": [
                        {
                            "pass": idx + 1,
                            "raw_vocabulary_size": step.raw_vocabulary_size,
                            "min_count": step.min_count,
                            "threshold": step.threshold,
                            "phrases_formed": step.phrases_formed,
                            "phrases_retained": step.phrases_retained,
                            "vocabulary_sha256": step.vocabulary_sha256,
                        }
                        for idx, step in enumerate(phrase.steps)
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


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
