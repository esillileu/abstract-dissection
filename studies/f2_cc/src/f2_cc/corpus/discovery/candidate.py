"""Candidate record model for Common Crawl probability and seed discovery."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


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
