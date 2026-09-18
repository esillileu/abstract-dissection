# Corpus substitution plan

The original Google News, LDC and large phrase corpora are unavailable or
insufficiently identified. Substitute-corpus experiments therefore compare
every verified corpus that can satisfy a condition; they do not select one
preferred substitute and discard the others.

## Comparison policy

For each paper condition:

1. enumerate every verified normalized corpus with enough lexical words;
2. materialize the requested lexical-token budget as an ordered prefix;
3. run the same model condition and seeds independently for every eligible corpus;
4. preserve corpus resource version, ordered-manifest digest and materialized
   corpus digest in the planned slot and MLflow lineage;
5. report each corpus separately and compare corpus-domain sensitivity across
   corpora; and
6. label every result as a reconstruction, never as an exact reproduction.

The existing ordered-manifest materializer streams shards in immutable order and
stops at the exact lexical-token budget. Larger corpora can therefore supply
smaller conditions without another stored corpus version. A smaller corpus must
not be repeated, concatenated with itself or relabeled to satisfy a larger
condition.

## Available normalized corpora

| resource version | domain | lexical words | supported standard budgets |
|---|---|---:|---|
| `f2-wmt-news-2007-2012-normalized-v1` | news | 1,646,423,517 | 24M, 49M, 50M, 98M, 196M, 320M, 391M, 783M, 1B, 1.6B |
| `f2-lm1b-r13output-normalized-v1` | language-model benchmark/news | 791,844,834 | 24M, 49M, 50M, 98M, 196M, 320M, 391M, 783M |
| `f2-umbc-webbase-normalized-v1` | web | 3,398,076,749 | 24M, 49M, 50M, 98M, 196M, 320M, 391M, 783M, 1B, 1.6B |

All three are connected to applicable catalog requirements through
`requirement_candidates`. A candidate records a reviewed possibility; only an
execution-plan binding creates runnable slots.

## Experiment coverage

| catalog condition | WMT | LM1B | UMBC | action |
|---|---|---|---|---|
| W2V1 Table 2, 24M-783M | full | full | full | Run all three corpora and compare all dimensions/seeds. |
| W2V1 Table 3, 320M | full | full | full | Run all three after non-corpus blockers are resolved. |
| W2V1 Table 4/5, 783M | full | full | full | Run all three. |
| W2V1 Table 4/5, 1.6B | full | insufficient | full | Run WMT and UMBC; report LM1B only as a separate 791,844,834-word reduced condition if desired. |
| W2V1 sentence training, 50M | full | full | full | Run all three as explicitly domain-substituted training conditions. |
| W2V2 word/phrase, 1B | full | insufficient | full | Run WMT and UMBC; optionally compare the labeled LM1B reduced condition. |
| W2V1 6B | insufficient | insufficient | insufficient | Corpus-blocked. |
| W2V2 6B/33B | insufficient | insufficient | insufficient | Corpus-blocked. |

Partial LM1B conditions must have their actual 791,844,834-word budget in their
slot identity and reports. They must not occupy a nominal 1B or 1.6B slot.

## Current executability

Training code exists today for these WMT-bound plans:

- W2V1 Table 2 CBOW: 24 conditions × seeds 1, 7 and 19;
- W2V2 1B phrase Skip-gram: six objective/subsampling conditions × the same seeds.

They can start through the tracked runner after F2 database, corpus S3 and MLflow
preflight and explicit large-run approval. Training ends after durable checkpoint,
lookup, observation and lineage publication; it does not resolve or read an
evaluation dataset. Full paper evaluation remains a later, independent operation
against the saved lookup artifact.

Corpus-specific execution-plan bindings, immutable slot IDs and runtime configs
for WMT, LM1B and UMBC are now materialized. The catalog contains 270 slots:
216 W2V1 slots and 54 W2V2 slots. LM1B W2V2 uses a separate 791,844,834-word
reduced plan and is never placed in the nominal 1B plan.

The remaining training gate is external: the corpus S3/DB/MLflow preflight must
pass and `--approve-large-run` must be supplied. Full paper evaluation still
needs the non-corpus resources listed separately. Checked-in question fixtures
are used only to test the standalone artifact evaluator, not canonical training.
