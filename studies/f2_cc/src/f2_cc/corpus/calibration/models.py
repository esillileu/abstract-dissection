"""Data models and result schemas for calibration, ablation, and recommendations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostFetchOperatingPoint:
    threshold: float
    doc_precision: float
    doc_recall: float
    word_precision: float
    word_recall: float
    byte_precision: float
    byte_recall: float
    storage_savings_pct: float
    is_cross_validated: bool = False


@dataclass(frozen=True)
class RuleAblationResult:
    rule_name: str
    description: str
    hits_count: int
    hits_pct: float
    bytes_saved: int
    bytes_saved_pct: float
    proxy_words_in_reject: int
    proxy_words_loss_pct: float
    sample_valid_news_count: int


@dataclass(frozen=True)
class ProductionPipelineRecommendation:
    config_name: str
    prefetch_rule_desc: str
    postfetch_threshold: float
    joint_word_recall: float
    joint_doc_recall: float
    joint_word_ppv: float
    joint_doc_ppv: float
    avoided_arc_requests_pct: float
    avoided_network_bytes_pct: float
    avoided_disk_storage_pct: float
    projected_net_words_15pct_dedup: float
    projected_net_words_50pct_dedup: float
    margin_vs_33b_50pct_dedup: float


__all__ = [
    "PostFetchOperatingPoint",
    "ProductionPipelineRecommendation",
    "RuleAblationResult",
]
