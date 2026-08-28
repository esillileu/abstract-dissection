-- Migration 002: Add pre-fetch rule tagging, reject exploration stream, and 8-stratum design tracking

ALTER TABLE candidate_records
    ADD COLUMN IF NOT EXISTS prefilter_status VARCHAR(32) NOT NULL DEFAULT 'pass',
    ADD COLUMN IF NOT EXISTS prefilter_rule VARCHAR(64) NOT NULL DEFAULT 'none',
    ADD COLUMN IF NOT EXISTS fetch_probability DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS is_selected_for_fetch BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE processing_results
    ADD COLUMN IF NOT EXISTS prefilter_status VARCHAR(32) NOT NULL DEFAULT 'pass',
    ADD COLUMN IF NOT EXISTS is_reject_exploration BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE audit_assignments
    ADD COLUMN IF NOT EXISTS design_stratum VARCHAR(64);

CREATE INDEX IF NOT EXISTS idx_candidates_prefetch ON candidate_records(run_id, prefilter_status, is_selected_for_fetch);
CREATE INDEX IF NOT EXISTS idx_audit_design_stratum ON audit_assignments(run_id, design_stratum, priority_order);
