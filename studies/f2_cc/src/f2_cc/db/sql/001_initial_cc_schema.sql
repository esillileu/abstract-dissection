CREATE TABLE pipeline_runs (
    run_id VARCHAR(64) PRIMARY KEY, run_type VARCHAR(32) NOT NULL, crawl_ids TEXT[] NOT NULL,
    sample_size INT NOT NULL, seed INT NOT NULL, bandwidth_mbps DOUBLE PRECISION NOT NULL,
    concurrency INT NOT NULL, status VARCHAR(32) NOT NULL DEFAULT 'running', output_dir TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), finished_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE candidate_records (
    candidate_id VARCHAR(64) NOT NULL, run_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs ON DELETE CASCADE,
    crawl_id VARCHAR(64) NOT NULL, url TEXT NOT NULL, url_timestamp VARCHAR(32) NOT NULL,
    arc_filename TEXT NOT NULL, arc_offset BIGINT NOT NULL, arc_length INT NOT NULL, arc_digest VARCHAR(64),
    source_type VARCHAR(32) NOT NULL, stratum VARCHAR(64), inclusion_probability DOUBLE PRECISION NOT NULL,
    design_weight DOUBLE PRECISION NOT NULL, block_index INT NOT NULL, record_index_in_block INT NOT NULL,
    block_total_records INT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prefilter_status VARCHAR(32) NOT NULL DEFAULT 'pass', prefilter_rule VARCHAR(64) NOT NULL DEFAULT 'none',
    fetch_probability DOUBLE PRECISION NOT NULL DEFAULT 1.0, is_selected_for_fetch BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY(run_id,candidate_id), UNIQUE(run_id,crawl_id,arc_filename,arc_offset,arc_length,url)
);
CREATE TABLE processing_results (
    candidate_id VARCHAR(64) NOT NULL, run_id VARCHAR(64) NOT NULL,
    fetch_status VARCHAR(32) NOT NULL, http_status INT NOT NULL DEFAULT 0, downloaded_bytes INT NOT NULL DEFAULT 0,
    extraction_success BOOLEAN NOT NULL DEFAULT FALSE, news_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    is_news_predicted BOOLEAN NOT NULL DEFAULT FALSE, is_english BOOLEAN NOT NULL DEFAULT FALSE,
    is_valid BOOLEAN NOT NULL DEFAULT FALSE, rejection_reason TEXT, word_count INT NOT NULL DEFAULT 0,
    word_count_proxy INT NOT NULL DEFAULT 0, clean_text_sha256 VARCHAR(64), shard_path TEXT,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
    prefilter_status VARCHAR(32) NOT NULL DEFAULT 'pass', is_reject_exploration BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY(run_id,candidate_id), FOREIGN KEY(run_id,candidate_id) REFERENCES candidate_records ON DELETE CASCADE
);
CREATE TABLE audit_assignments (
    audit_id VARCHAR(64) PRIMARY KEY, run_id VARCHAR(64) NOT NULL, candidate_id VARCHAR(64) NOT NULL,
    audit_stratum INT NOT NULL, priority_order INT NOT NULL, wave INT NOT NULL DEFAULT 1,
    audit_inclusion_probability DOUBLE PRECISION NOT NULL, audit_design_weight DOUBLE PRECISION NOT NULL,
    is_audited BOOLEAN NOT NULL DEFAULT FALSE, gold_class INT, word_count_gold INT, word_residual INT,
    audited_at TIMESTAMPTZ, auditor_id VARCHAR(64), notes TEXT, design_stratum VARCHAR(64),
    FOREIGN KEY(run_id,candidate_id) REFERENCES candidate_records ON DELETE CASCADE,
    UNIQUE(run_id,candidate_id)
);
CREATE TABLE analysis_profiles (
    profile_key VARCHAR(64) NOT NULL, revision INT NOT NULL, purpose VARCHAR(32) NOT NULL,
    run_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs ON DELETE RESTRICT, rationale TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(profile_key,revision), CHECK(purpose IN ('analysis','calibration'))
);
CREATE UNIQUE INDEX uq_cc_active_profile ON analysis_profiles(profile_key) WHERE is_active;
CREATE INDEX idx_cc_candidates_run ON candidate_records(run_id);
CREATE INDEX idx_cc_results_run ON processing_results(run_id);
CREATE INDEX idx_cc_audit_run ON audit_assignments(run_id,priority_order);
