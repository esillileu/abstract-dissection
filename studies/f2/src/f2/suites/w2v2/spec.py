"""Validated W2V2 word and phrase run specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from f2.suites.w2v.phrases import PhrasePolicy
from f2.suites.w2v.spec import Word2VecRunSpec
from repro_core.execution.spec import load_variant, mapping


@dataclass(frozen=True)
class RunSpec(Word2VecRunSpec):
    """Select word or phrase input without changing historical phrase identities."""

    phrase_detection: dict[str, object] = field(default_factory=dict)
    large_scale: dict[str, object] = field(default_factory=dict)
    corpus_mode: str = "phrase"
    subsampling_policy: str | None = None

    def resolved_config(self) -> dict[str, object]:
        resolved = super().resolved_config()
        if self.corpus_mode == "phrase":
            resolved["phrase_detection"] = self.phrase_detection
        else:
            resolved["corpus_mode"] = self.corpus_mode
            resolved["subsampling_policy"] = self.subsampling_policy
        if self.large_scale:
            resolved["large_scale"] = self.large_scale
        return resolved


def parse_run_spec(
    path: str | Path,
    *,
    atomic_run_id: str | None = None,
    overrides: dict[str, object] | None = None,
) -> RunSpec:
    path = Path(path)
    raw = load_variant(path, atomic_run_id=atomic_run_id, overrides=overrides)
    if raw.get("domain") != "f2.w2v2" or raw.get("kind") != "word2vec":
        raise ValueError("W2V2 config requires domain f2.w2v2 and kind word2vec")
    identity = mapping(raw, "identity")
    required = {
        "execution_plan_id",
        "planned_run_slot_id",
        "seed",
    }
    if missing := sorted(required - identity.keys()):
        raise ValueError(f"W2V2 identity is missing: {', '.join(missing)}")
    training = mapping(raw, "training")
    objective = training.get("objective_kind")
    if objective not in {"negative_sampling", "hierarchical_softmax", "nce"}:
        raise ValueError(f"unsupported W2V2 objective: {objective}")
    mode = raw.get("corpus_mode", "phrase")
    if mode not in {"word", "phrase"}:
        raise ValueError(f"unsupported W2V2 corpus_mode: {mode}")
    policy = raw.get("subsampling_policy")
    if policy is not None and policy != "word2vec_c":
        raise ValueError("W2V2 supports only subsampling_policy word2vec_c")
    if mode == "word" and policy is None:
        raise ValueError("word-only W2V2 requires explicit subsampling_policy")
    if mode == "word" and "phrase_detection" in raw:
        raise ValueError("word-only W2V2 cannot configure phrase_detection")
    phrase = mapping(raw, "phrase_detection") if mode == "phrase" else {}
    if mode == "phrase":
        PhrasePolicy.from_config(phrase)
    large = mapping(raw, "large_scale") if "large_scale" in raw else {}
    if large:
        from .preflight import SENTENCE_POLICY

        paper_tokens = large.get("paper_lexical_tokens")
        actual_tokens = large.get("lexical_tokens")
        if (
            paper_tokens not in {6_000_000_000, 33_000_000_000}
            or type(actual_tokens) is not int
            or not 0 < actual_tokens <= paper_tokens
            or (paper_tokens == 6_000_000_000 and actual_tokens != paper_tokens)
        ):
            raise ValueError("invalid Section 4.1 paper/actual lexical token budget")
        revision = 2 if actual_tokens == paper_tokens else 3
        if (
            large.get("sentence_policy") != SENTENCE_POLICY
            or large.get("artifact_id")
            != f"f2.w2v2.section4.phrase-{paper_tokens // 1_000_000_000}b.hs1000.entire-sentence.r{revision}"
            or mapping(raw, "corpus").get("lexical_token_budget")
            != large.get("lexical_tokens")
            or large.get("execution_policy") != "contiguous-sentence-shards-v2"
            or phrase.get("passes") != 2
            or phrase.get("separator", "_") != "_"
            or raw.get("distribution") is not None
            or mode != "phrase"
            or training.get("model_kind") != "skip_gram"
            or objective != "hierarchical_softmax"
            or training.get("embedding_dimension") != 1000
            or training.get("context_policy") != "entire_sentence"
        ):
            raise ValueError(
                "Section 4.1 requires 6B/33B phrase Skip-gram, 1000d HS and explicit sentence policy"
            )
    elif training.get("context_policy") == "entire_sentence":
        raise ValueError(
            "entire_sentence requires explicit large_scale reconstruction policy"
        )
    return RunSpec(
        atomic_run_id=str(raw["atomic_run_id"]),
        identity=identity,
        corpus=mapping(raw, "corpus"),
        vocabulary=mapping(raw, "vocabulary"),
        training=training,
        checkpoint=mapping(raw, "checkpoint"),
        tracking=mapping(raw, "tracking"),
        path=path,
        phrase_detection=phrase,
        large_scale=large,
        corpus_mode=str(mode),
        subsampling_policy=policy,
    )


__all__ = ["RunSpec", "parse_run_spec"]
