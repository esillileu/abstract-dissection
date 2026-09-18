"""Phrase policy layered on the shared W2V epoch executor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.evaluation import (
    additive_composition,
    evaluate_analogies,
    nearest_tokens,
    parse_analogy_questions,
    pca_projection,
)
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
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        expected = str(corpus.get("sha256", source_digest))
        if source_digest != expected:
            raise ValueError("W2V2 source corpus identity mismatch")
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
        )
        resolved["corpus"] = {"path": str(phrase.path), "sha256": phrase.corpus_sha256}
        result = super().run(resolved, context)
        lineage_target = result.root / "phrase_lineage.json"
        lineage_target.write_bytes(phrase.path.with_suffix(".txt.json").read_bytes())
        self._write_phrase_report(resolved, result, context, policy)
        return result

    @staticmethod
    def _write_phrase_report(
        config: dict[str, object],
        result: W2V1Result,
        context: ExperimentContext,
        policy: PhrasePolicy,
    ) -> None:
        lookup = load_lookup_artifact(result.lookup)
        evaluation = _mapping(config, "evaluation")
        questions_path = Path(str(evaluation["questions_path"]))
        if not questions_path.is_absolute():
            questions_path = context.paths.repo_root / questions_path
        analogy = evaluate_analogies(
            lookup, parse_analogy_questions(questions_path.read_bytes().splitlines())
        )
        phrase_tokens = [
            bytes(
                lookup.token_bytes[
                    lookup.token_offsets[row] : lookup.token_offsets[row + 1]
                ]
            )
            for row in range(lookup.counts.size)
            if policy.separator
            in bytes(
                lookup.token_bytes[
                    lookup.token_offsets[row] : lookup.token_offsets[row + 1]
                ]
            )
        ][:8]
        neighbors = {
            token.hex(): [
                asdict(item) | {"token": item.token.hex()}
                for item in nearest_tokens(lookup, token, limit=5)
            ]
            for token in phrase_tokens
        }
        composition_tokens = phrase_tokens[:2]
        composition = [
            asdict(item) | {"token": item.token.hex()}
            for item in additive_composition(lookup, composition_tokens, limit=5)
        ]
        projection = {
            token.hex(): point
            for token, point in pca_projection(lookup, phrase_tokens).items()
        }
        payload = {
            "analogy": {
                "overall": asdict(analogy.overall),
                "categories": [asdict(category) for category in analogy.categories],
            },
            "nearest_entities": neighbors,
            "additive_composition": {
                "tokens": [token.hex() for token in composition_tokens],
                "neighbors": composition,
            },
            "pca": projection,
            "token_encoding": "hex",
        }
        path = result.root / "phrase_evaluation.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        report = json.loads(result.report.read_text())
        report["phrase_evaluation"] = "phrase_evaluation.json"
        report["artifacts"]["phrase_lineage"] = "phrase_lineage.json"
        result.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


EXECUTORS = {"word2vec": W2V2Executor()}


def get_executor(kind: str) -> W2V2Executor:
    try:
        return EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown W2V2 experiment kind: {kind}") from exc


__all__ = ["W2V2Executor", "get_executor"]
