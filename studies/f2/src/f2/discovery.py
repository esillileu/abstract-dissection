"""Candidate discovery, seed domain catalog, crawl-stratified two-stage probability sampling, and sequential audit sampling."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .cdx import CDXBlockLocator, CDXIndexReader, CDXRecord


class DomainStratum(StrEnum):
    GLOBAL = "global_agencies"
    NATIONAL = "national_press"
    REGIONAL = "regional_press"
    SPECIALTY = "specialty_media"
    SYNDICATED = "syndicated_outlets"


SEED_DOMAIN_CATALOG: dict[DomainStratum, list[str]] = {
    DomainStratum.GLOBAL: [
        "reuters.com",
        "bbc.co.uk",
        "ap.org",
        "bloomberg.com",
        "afp.com",
    ],
    DomainStratum.NATIONAL: [
        "nytimes.com",
        "washingtonpost.com",
        "theguardian.com",
        "telegraph.co.uk",
        "smh.com.au",
        "wsj.com",
        "usatoday.com",
        "independent.co.uk",
    ],
    DomainStratum.REGIONAL: [
        "chicagotribune.com",
        "sfgate.com",
        "seattletimes.com",
        "boston.com",
        "latimes.com",
        "startribune.com",
        "denverpost.com",
    ],
    DomainStratum.SPECIALTY: [
        "techcrunch.com",
        "wired.com",
        "arstechnica.com",
        "economist.com",
        "forbes.com",
        "cnet.com",
        "venturebeat.com",
    ],
    DomainStratum.SYNDICATED: [
        "prnewswire.com",
        "businesswire.com",
        "marketwatch.com",
        "upi.com",
    ],
}

NEWS_PATH_PATTERNS = [
    re.compile(r"/(?:news|article|articles|story|stories)/", re.IGNORECASE),
    re.compile(
        r"/(?:world|politics|business|technology|science|opinion)/\d{4}/", re.IGNORECASE
    ),
    re.compile(r"/\d{4}/\d{1,2}/\d{1,2}/[a-z0-9\-]+", re.IGNORECASE),
    re.compile(r"/[a-z0-9\-]+-\d{5,}\.html?$", re.IGNORECASE),
]


def is_news_path_heuristic(url: str) -> bool:
    """Evaluate whether URL matches standard news path heuristics."""
    return any(pattern.search(url) is not None for pattern in NEWS_PATH_PATTERNS)


@dataclass(frozen=True)
class CandidateRecord:
    crawl_id: str
    url: str
    timestamp: str
    filename: str
    offset: int
    length: int
    digest: str
    source_type: str  # "probability_sample" or "seed_catalog"
    stratum: str | None
    inclusion_probability: float
    design_weight: float
    block_index: int
    record_index_in_block: int
    block_total_records: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_id(self) -> str:
        key = f"{self.crawl_id}:{self.filename}:{self.offset}:{self.length}:{self.url}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


class TwoStageProbabilitySampler:
    """Draws a rigorous two-stage probability sample from Common Crawl CDX cluster index."""

    def __init__(
        self, crawl_id: str, index_reader: CDXIndexReader, seed: int = 42
    ) -> None:
        self.crawl_id = crawl_id
        self.reader = index_reader
        self.seed = seed
        self.rng = random.Random(seed)

    def plan_stage1_blocks(self, num_blocks: int) -> list[CDXBlockLocator]:
        """Select m primary sampling units (CDX blocks) via SRS without replacement."""
        total_blocks = self.reader.total_blocks()
        m = min(num_blocks, total_blocks)
        indices = sorted(self.rng.sample(range(total_blocks), m))
        return [self.reader.entries[i] for i in indices]

    def sample_block_records(
        self,
        block: CDXBlockLocator,
        records: list[CDXRecord],
        num_records_per_block: int,
    ) -> list[CandidateRecord]:
        """Stage 2: Select n_k records from decompressed block records (M_k) and compute exact pi_ki."""
        M_k = len(records)
        if M_k == 0:
            return []

        n_k = min(num_records_per_block, M_k)
        # Record inclusion probability: pi_ki = p_k * (n_k / M_k)
        # where p_k = m / K
        p_k = 1.0  # Will be multiplied by stage 1 probability at assembly

        # Sample record indices deterministically using block-seeded RNG
        block_seed = int(
            hashlib.md5(f"{self.seed}:{block.block_index}".encode()).hexdigest()[:8], 16
        )
        block_rng = random.Random(block_seed)
        sampled_indices = sorted(block_rng.sample(range(M_k), n_k))

        sampled: list[CandidateRecord] = []
        for idx in sampled_indices:
            rec = records[idx]
            # Conditional probability given block selection:
            pi_within = n_k / M_k
            sampled.append(
                CandidateRecord(
                    crawl_id=self.crawl_id,
                    url=rec.url,
                    timestamp=rec.timestamp,
                    filename=rec.filename,
                    offset=rec.offset,
                    length=rec.length,
                    digest=rec.digest,
                    source_type="probability_sample",
                    stratum="unbiased_crawl_wide",
                    inclusion_probability=pi_within,  # Stage 1 factor applied in finalize
                    design_weight=1.0 / pi_within,
                    block_index=block.block_index,
                    record_index_in_block=idx,
                    block_total_records=M_k,
                )
            )
        return sampled

    def finalize_inclusion_probabilities(
        self,
        candidates: list[CandidateRecord],
        num_selected_blocks: int,
        total_crawl_blocks: int,
    ) -> list[CandidateRecord]:
        """Apply stage 1 block selection probability p_k = m / K to all sampled records."""
        p_k = num_selected_blocks / total_crawl_blocks
        finalized: list[CandidateRecord] = []
        for c in candidates:
            full_pi = p_k * c.inclusion_probability
            weight = 1.0 / full_pi if full_pi > 0 else 0.0
            finalized.append(
                CandidateRecord(
                    crawl_id=c.crawl_id,
                    url=c.url,
                    timestamp=c.timestamp,
                    filename=c.filename,
                    offset=c.offset,
                    length=c.length,
                    digest=c.digest,
                    source_type=c.source_type,
                    stratum=c.stratum,
                    inclusion_probability=full_pi,
                    design_weight=weight,
                    block_index=c.block_index,
                    record_index_in_block=c.record_index_in_block,
                    block_total_records=c.block_total_records,
                    metadata=c.metadata,
                )
            )
        return finalized


class SequentialAuditSampler:
    """Pre-specified sequential probability audit sampler with exact permutation priority."""

    def __init__(self, seed: int = 1337) -> None:
        self.seed = seed

    def generate_audit_schedule(
        self,
        candidates: list[CandidateRecord],
        predicted_classes: list[int],
    ) -> list[dict[str, Any]]:
        """Assign pre-specified random priority permutation per stratum so extension preserves exact SRS."""
        strata_buckets: dict[int, list[tuple[CandidateRecord, int]]] = {0: [], 1: []}
        for cand, pred in zip(candidates, predicted_classes, strict=False):
            strata_buckets[pred].append((cand, pred))

        rng = random.Random(self.seed)
        schedule: list[dict[str, Any]] = []

        for _pred_class, items in strata_buckets.items():
            shuffled_indices = list(range(len(items)))
            rng.shuffle(shuffled_indices)
            for priority, orig_idx in enumerate(shuffled_indices):
                cand, pred = items[orig_idx]
                schedule.append(
                    {
                        "record_id": cand.record_id(),
                        "url": cand.url,
                        "crawl_id": cand.crawl_id,
                        "predicted_class": pred,
                        "stratum_total": len(items),
                        "priority_order": priority,  # 0, 1, 2...
                    }
                )
        return schedule

    @staticmethod
    def select_audit_wave(
        audit_schedule: list[dict[str, Any]],
        num_per_stratum: dict[int, int],
    ) -> list[dict[str, Any]]:
        """Select audit units up to the allocated stratum budget using pre-specified priority."""
        selected: list[dict[str, Any]] = []
        for item in audit_schedule:
            pred = item["predicted_class"]
            limit = num_per_stratum.get(pred, 0)
            if item["priority_order"] < limit:
                # Conditional audit inclusion probability
                pi_audit_cond = (
                    limit / item["stratum_total"] if item["stratum_total"] > 0 else 0.0
                )
                augmented = dict(item)
                augmented["audit_inclusion_prob_cond"] = pi_audit_cond
                augmented["audit_weight_cond"] = (
                    1.0 / pi_audit_cond if pi_audit_cond > 0 else 0.0
                )
                selected.append(augmented)
        return selected


__all__ = [
    "NEWS_PATH_PATTERNS",
    "SEED_DOMAIN_CATALOG",
    "CandidateRecord",
    "DomainStratum",
    "SequentialAuditSampler",
    "TwoStageProbabilitySampler",
    "is_news_path_heuristic",
]
