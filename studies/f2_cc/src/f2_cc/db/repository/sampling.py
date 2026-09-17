"""Candidate insertion and document processing state recording."""

from __future__ import annotations

import json

from f2_cc.corpus.discovery import CandidateRecord
from f2_cc.corpus.pipeline import ProcessedDocumentResult

from .base import BaseRepository


class SamplingRepositoryMixin(BaseRepository):
    """Operations for recording candidate records and processing results."""

    def insert_candidates(self, run_id: str, candidates: list[CandidateRecord]) -> int:
        if not candidates:
            return 0
        count = 0
        with self.conn.cursor() as cur:
            for c in candidates:
                cand_id = c.record_id()
                cur.execute(
                    """
                    INSERT INTO candidate_records (
                        candidate_id, run_id, crawl_id, url, url_timestamp,
                        arc_filename, arc_offset, arc_length, arc_digest,
                        source_type, stratum, inclusion_probability, design_weight,
                        block_index, record_index_in_block, block_total_records,
                        prefilter_status, prefilter_rule, fetch_probability, is_selected_for_fetch
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, candidate_id) DO NOTHING;
                    """,
                    (
                        cand_id,
                        run_id,
                        c.crawl_id,
                        c.url,
                        c.timestamp,
                        c.filename,
                        c.offset,
                        c.length,
                        c.digest,
                        c.source_type,
                        c.stratum,
                        c.inclusion_probability,
                        c.design_weight,
                        c.block_index,
                        c.record_index_in_block,
                        c.block_total_records,
                        c.prefilter_status,
                        c.prefilter_rule,
                        c.fetch_probability,
                        c.is_selected_for_fetch,
                    ),
                )
                count += cur.rowcount
        self._commit()
        return count

    def get_completed_candidate_ids(self, run_id: str) -> set[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT candidate_id FROM processing_results
                WHERE run_id = %s;
                """,
                (run_id,),
            )
            return {row[0] for row in cur.fetchall()}

    def record_processing_result(
        self,
        run_id: str,
        result: ProcessedDocumentResult,
        clean_text_sha256: str | None = None,
        shard_path: str | None = None,
        prefilter_status: str = "pass",
        is_reject_exploration: bool = False,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processing_results (
                    candidate_id, run_id, fetch_status, http_status,
                    downloaded_bytes, extraction_success, news_score,
                    is_news_predicted, is_english, is_valid, rejection_reason,
                    word_count, word_count_proxy, clean_text_sha256, shard_path,
                    prefilter_status, is_reject_exploration, diagnostics
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, candidate_id) DO UPDATE SET
                    fetch_status = EXCLUDED.fetch_status,
                    http_status = EXCLUDED.http_status,
                    downloaded_bytes = EXCLUDED.downloaded_bytes,
                    extraction_success = EXCLUDED.extraction_success,
                    news_score = EXCLUDED.news_score,
                    is_news_predicted = EXCLUDED.is_news_predicted,
                    is_english = EXCLUDED.is_english,
                    is_valid = EXCLUDED.is_valid,
                    rejection_reason = EXCLUDED.rejection_reason,
                    word_count = EXCLUDED.word_count,
                    word_count_proxy = EXCLUDED.word_count_proxy,
                    clean_text_sha256 = EXCLUDED.clean_text_sha256,
                    shard_path = EXCLUDED.shard_path,
                    prefilter_status = EXCLUDED.prefilter_status,
                    is_reject_exploration = EXCLUDED.is_reject_exploration,
                    diagnostics = EXCLUDED.diagnostics,
                    processed_at = NOW();
                """,
                (
                    result.record_id,
                    run_id,
                    result.fetch_status,
                    result.http_status,
                    result.downloaded_bytes,
                    result.extraction_success,
                    result.news_score,
                    result.is_news_predicted,
                    result.is_english,
                    result.is_valid,
                    result.rejection_reason,
                    result.word_count,
                    result.proxy_words,
                    clean_text_sha256,
                    shard_path,
                    prefilter_status,
                    is_reject_exploration,
                    json.dumps(result.diagnostics),
                ),
            )
        self._commit()
