from __future__ import annotations

ATOMIC_RUN_IDS = (
    "W2V-PTB-CBOW-NS",
    "W2V-PTB-SKIPGRAM-NS",
    "W2V-PTB-CBOW-FULL",
    "W2V-PTB-SKIPGRAM-FULL",
)
CURVE_ATOMIC_RUN_IDS = (
    "W2V-PTB-CBOW-NS",
    "W2V-PTB-SKIPGRAM-NS",
)
ORIGINAL_NATIVE_IDS = {
    "W2V-PTB-CBOW-NS": "PTB-CBOW",
    "W2V-PTB-SKIPGRAM-NS": "PTB-SKIPGRAM",
}
SIMILARITY_QUERIES = ("you", "year", "car", "toyota")
ANALOGY_QUERIES = (
    ("king", "man", "queen", "woman"),
    ("take", "took", "go", "went"),
    ("car", "cars", "child", "children"),
    ("good", "better", "bad", "worse"),
)
TOP_K = 5
CSV_FIELDS = (
    "series",
    "seed",
    "run_id",
    "task",
    "query",
    "expected",
    "expected_rank",
    "hit_at_5",
    "candidate_rank",
    "candidate",
    "score",
)
