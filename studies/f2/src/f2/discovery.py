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

# Exact 32-extension Rule 1 regex from 10k calibration baseline
ALL_BINARY_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|css|js|pdf|mp3|mp4|avi|zip|gz|tar|tgz|exe|dmg|iso|bin|doc|docx|ppt|pptx|xls|xlsx|rss|xml|json|swf|ico|woff|ttf|svg)(\?.*)?$",
    re.IGNORECASE,
)


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
    prefilter_status: str = "pass"  # "pass" or "reject"
    prefilter_rule: str = "none"  # "rule1" or "none"
    fetch_probability: float = 1.0  # 1.0 for pass, 0.05 for reject exploration
    is_selected_for_fetch: bool = True
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
        # Sample record indices deterministically using block-seeded RNG
        block_seed = int(
            hashlib.md5(f"{self.seed}:{block.block_index}".encode()).hexdigest()[:8], 16
        )
        block_rng = random.Random(block_seed)
        sampled_indices = sorted(block_rng.sample(range(M_k), n_k))

        sampled: list[CandidateRecord] = []
        for idx in sampled_indices:
            rec = records[idx]
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
        prefetch_rule: str = "none",
        reject_exploration_rate: float = 0.05,
    ) -> list[CandidateRecord]:
        """Apply stage 1 block selection probability p_k = m / K and optional pre-fetch reject exploration."""
        p_k = num_selected_blocks / total_crawl_blocks
        finalized: list[CandidateRecord] = []

        for c in candidates:
            base_pi = p_k * c.inclusion_probability

            # Pre-fetch evaluation
            is_reject = False
            if prefetch_rule == "rule1":
                is_reject = bool(ALL_BINARY_EXT.search(c.url))

            if is_reject:
                pref_status = "reject"
                pref_rule = "rule1"
                fetch_prob = reject_exploration_rate
                # Deterministic PRNG draw for reject exploration based on candidate identity & seed
                cand_seed = int(
                    hashlib.md5(
                        f"{self.seed}:reject:{c.record_id()}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                cand_rng = random.Random(cand_seed)
                is_selected = cand_rng.random() < fetch_prob
            else:
                pref_status = "pass"
                pref_rule = "rule1" if prefetch_rule == "rule1" else "none"
                fetch_prob = 1.0
                is_selected = True

            total_pi = base_pi * fetch_prob
            weight = 1.0 / total_pi if total_pi > 0 else 0.0

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
                    inclusion_probability=base_pi,
                    design_weight=weight,
                    block_index=c.block_index,
                    record_index_in_block=c.record_index_in_block,
                    block_total_records=c.block_total_records,
                    prefilter_status=pref_status,
                    prefilter_rule=pref_rule,
                    fetch_probability=fetch_prob,
                    is_selected_for_fetch=is_selected,
                    metadata=c.metadata,
                )
            )
        return finalized


class SequentialAuditSampler:
    """Pre-specified sequential probability audit sampler with exact permutation priority and 8-stratum support."""

    def __init__(self, seed: int = 1337) -> None:
        self.seed = seed

    @staticmethod
    def get_stratum_id(
        crawl_id: str, prefilter_status: str, is_news_predicted: bool | int
    ) -> str:
        """Map record attributes to one of the 8 canonical design strata (S1 to S8)."""
        is_09_10 = "2009-2010" in str(crawl_id)
        is_pass = str(prefilter_status).lower() == "pass"
        is_pos = int(is_news_predicted) == 1

        if is_09_10:
            if is_pass:
                return "S1" if is_pos else "S2"
            else:
                return "S3" if is_pos else "S4"
        else:
            if is_pass:
                return "S5" if is_pos else "S6"
            else:
                return "S7" if is_pos else "S8"

    def generate_8_stratum_audit_schedule(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Partition candidates across 8 design strata and assign random priority permutations within each stratum."""
        strata_buckets: dict[str, list[dict[str, Any]]] = {
            f"S{i}": [] for i in range(1, 9)
        }

        for r in records:
            crawl = r.get("crawl_id", "")
            pref = r.get("prefilter_status", "pass")
            pred = r.get("is_news_predicted", 0)
            sid = self.get_stratum_id(crawl, pref, pred)
            strata_buckets[sid].append(r)

        rng = random.Random(self.seed)
        schedule: list[dict[str, Any]] = []

        for sid in sorted(strata_buckets.keys()):
            items = strata_buckets[sid]
            shuffled_indices = list(range(len(items)))
            rng.shuffle(shuffled_indices)
            for priority, orig_idx in enumerate(shuffled_indices):
                rec = items[orig_idx]
                schedule.append(
                    {
                        "candidate_id": rec.get("candidate_id") or rec.get("record_id"),
                        "record_id": rec.get("candidate_id") or rec.get("record_id"),
                        "url": rec["url"],
                        "crawl_id": rec["crawl_id"],
                        "prefilter_status": rec.get("prefilter_status", "pass"),
                        "is_news_predicted": int(rec.get("is_news_predicted", 0)),
                        "design_stratum": sid,
                        "stratum_total": len(items),
                        "priority_order": priority,
                    }
                )
        return schedule

    @staticmethod
    def select_8_stratum_audit_wave(
        schedule: list[dict[str, Any]],
        base_targets: dict[str, int] | None = None,
        total_budget: int = 400,
    ) -> list[dict[str, Any]]:
        """Select audit wave allocating n_h = min(T_h, N_h) and deterministically redistributing unused quota."""
        default_targets: dict[str, int] = {
            "S1": 150,
            "S2": 30,
            "S3": 10,
            "S4": 10,
            "S5": 150,
            "S6": 30,
            "S7": 10,
            "S8": 10,
        }
        targets = dict(base_targets or default_targets)

        # Calculate stratum sizes in schedule
        stratum_counts: dict[str, int] = {f"S{i}": 0 for i in range(1, 9)}
        for item in schedule:
            sid = item["design_stratum"]
            stratum_counts[sid] = item["stratum_total"]

        # Initial allocation n_h = min(T_h, N_h)
        allocated: dict[str, int] = {}
        unused_budget = 0
        for sid in sorted(stratum_counts.keys()):
            n_h = min(targets[sid], stratum_counts[sid])
            allocated[sid] = n_h
            unused_budget += max(0, targets[sid] - n_h)

        # Deterministic redistribution of unused quota:
        # Tier 1: To companion reject cell in same crawl (S3<->S4, S7<->S8)
        companion_pairs = [("S3", "S4"), ("S4", "S3"), ("S7", "S8"), ("S8", "S7")]
        for src, dst in companion_pairs:
            deficit = max(0, targets[src] - stratum_counts[src])
            if deficit > 0 and unused_budget > 0:
                capacity = stratum_counts[dst] - allocated[dst]
                transfer = min(deficit, capacity, unused_budget)
                if transfer > 0:
                    allocated[dst] += transfer
                    unused_budget -= transfer

        # Tier 2: To pass stream in same crawl (80% pos, 20% neg)
        crawl_pass_pos = ["S1", "S5"]
        crawl_pass_neg = ["S2", "S6"]
        for sid in crawl_pass_pos:
            if unused_budget > 0:
                cap = stratum_counts[sid] - allocated[sid]
                trans = min(cap, int(unused_budget * 0.8))
                allocated[sid] += trans
                unused_budget -= trans
        for sid in crawl_pass_neg:
            if unused_budget > 0:
                cap = stratum_counts[sid] - allocated[sid]
                trans = min(cap, unused_budget)
                allocated[sid] += trans
                unused_budget -= trans

        # Tier 3: Sequential across remaining capacity
        for sid in sorted(stratum_counts.keys()):
            if unused_budget <= 0:
                break
            cap = stratum_counts[sid] - allocated[sid]
            trans = min(cap, unused_budget)
            allocated[sid] += trans
            unused_budget -= trans

        selected: list[dict[str, Any]] = []
        for item in schedule:
            sid = item["design_stratum"]
            limit = allocated.get(sid, 0)
            if item["priority_order"] < limit:
                N_h = stratum_counts[sid]
                pi_audit_cond = limit / N_h if N_h > 0 else 0.0
                augmented = dict(item)
                augmented["audit_inclusion_prob_cond"] = pi_audit_cond
                augmented["audit_weight_cond"] = (
                    1.0 / pi_audit_cond if pi_audit_cond > 0 else 0.0
                )
                augmented["audit_stratum"] = 1 if item["is_news_predicted"] == 1 else 0
                selected.append(augmented)
        return selected

    def generate_audit_schedule(
        self,
        candidates: list[CandidateRecord | dict[str, Any]],
        predicted_classes: list[int],
    ) -> list[dict[str, Any]]:
        """Legacy 2-stratum schedule generator."""
        strata_buckets: dict[
            int, list[tuple[CandidateRecord | dict[str, Any], int]]
        ] = {0: [], 1: []}
        for cand, pred in zip(candidates, predicted_classes, strict=False):
            strata_buckets[pred].append((cand, pred))

        rng = random.Random(self.seed)
        schedule: list[dict[str, Any]] = []

        for _pred_class, items in strata_buckets.items():
            shuffled_indices = list(range(len(items)))
            rng.shuffle(shuffled_indices)
            for priority, orig_idx in enumerate(shuffled_indices):
                cand, pred = items[orig_idx]
                rec_id = (
                    cand["record_id"] if isinstance(cand, dict) else cand.record_id()
                )
                url = cand["url"] if isinstance(cand, dict) else cand.url
                crawl_id = cand["crawl_id"] if isinstance(cand, dict) else cand.crawl_id
                schedule.append(
                    {
                        "record_id": rec_id,
                        "url": url,
                        "crawl_id": crawl_id,
                        "predicted_class": pred,
                        "stratum_total": len(items),
                        "priority_order": priority,
                    }
                )
        return schedule

    @staticmethod
    def select_audit_wave(
        audit_schedule: list[dict[str, Any]],
        num_per_stratum: dict[int, int],
    ) -> list[dict[str, Any]]:
        """Legacy 2-stratum selector."""
        selected: list[dict[str, Any]] = []
        for item in audit_schedule:
            pred = item.get("predicted_class", item.get("audit_stratum", 0))
            limit = num_per_stratum.get(pred, 0)
            if item["priority_order"] < limit:
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
    "ALL_BINARY_EXT",
    "NEWS_PATH_PATTERNS",
    "SEED_DOMAIN_CATALOG",
    "CandidateRecord",
    "DomainStratum",
    "SequentialAuditSampler",
    "TwoStageProbabilitySampler",
    "is_news_path_heuristic",
]
