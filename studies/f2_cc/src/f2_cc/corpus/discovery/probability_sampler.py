"""Two-stage probability sampling from Common Crawl CDX cluster index."""

from __future__ import annotations

import hashlib
import random

from repro_io.commoncrawl.cdx import CDXBlockLocator, CDXIndexReader, CDXRecord

from .candidate import CandidateRecord
from .catalog import ALL_BINARY_EXT


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
