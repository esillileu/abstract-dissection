-- Migration 001: Initial F2 Word2Vec Corpus Operational State Schema

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(64) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    run_type VARCHAR(32) NOT NULL,
    crawl_ids TEXT[] NOT NULL,
    sample_size INT NOT NULL,
    seed INT NOT NULL,
    bandwidth_mbps DOUBLE PRECISION NOT NULL,
    concurrency INT NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'running',
    output_dir TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS candidate_records (
    candidate_id VARCHAR(64) NOT NULL,
    run_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    crawl_id VARCHAR(64) NOT NULL,
    url TEXT NOT NULL,
    url_timestamp VARCHAR(32) NOT NULL,
    arc_filename TEXT NOT NULL,
    arc_offset BIGINT NOT NULL,
    arc_length INT NOT NULL,
    arc_digest VARCHAR(64),
    source_type VARCHAR(32) NOT NULL,
    stratum VARCHAR(64),
    inclusion_probability DOUBLE PRECISION NOT NULL,
    design_weight DOUBLE PRECISION NOT NULL,
    block_index INT NOT NULL,
    record_index_in_block INT NOT NULL,
    block_total_records INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, candidate_id),
    CONSTRAINT uq_candidate_run_locator UNIQUE (run_id, crawl_id, arc_filename, arc_offset, arc_length, url)
);

CREATE TABLE IF NOT EXISTS processing_results (
    candidate_id VARCHAR(64) NOT NULL,
    run_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    fetch_status VARCHAR(32) NOT NULL,
    http_status INT NOT NULL DEFAULT 0,
    downloaded_bytes INT NOT NULL DEFAULT 0,
    extraction_success BOOLEAN NOT NULL DEFAULT FALSE,
    news_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    is_news_predicted BOOLEAN NOT NULL DEFAULT FALSE,
    is_english BOOLEAN NOT NULL DEFAULT FALSE,
    is_valid BOOLEAN NOT NULL DEFAULT FALSE,
    rejection_reason TEXT,
    word_count INT NOT NULL DEFAULT 0,
    word_count_proxy INT NOT NULL DEFAULT 0,
    clean_text_sha256 VARCHAR(64),
    shard_path TEXT,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (run_id, candidate_id),
    FOREIGN KEY (run_id, candidate_id) REFERENCES candidate_records(run_id, candidate_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_assignments (
    audit_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    candidate_id VARCHAR(64) NOT NULL,
    audit_stratum INT NOT NULL,
    priority_order INT NOT NULL,
    wave INT NOT NULL DEFAULT 1,
    audit_inclusion_probability DOUBLE PRECISION NOT NULL,
    audit_design_weight DOUBLE PRECISION NOT NULL,
    is_audited BOOLEAN NOT NULL DEFAULT FALSE,
    gold_class INT,
    word_count_gold INT,
    word_residual INT,
    audited_at TIMESTAMPTZ,
    auditor_id VARCHAR(64),
    notes TEXT,
    FOREIGN KEY (run_id, candidate_id) REFERENCES candidate_records(run_id, candidate_id) ON DELETE CASCADE,
    CONSTRAINT uq_audit_run_candidate UNIQUE (run_id, candidate_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_candidates_run ON candidate_records(run_id);
CREATE INDEX IF NOT EXISTS idx_candidates_crawl ON candidate_records(crawl_id);
CREATE INDEX IF NOT EXISTS idx_results_run ON processing_results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_status ON processing_results(fetch_status, is_valid, is_news_predicted);
CREATE INDEX IF NOT EXISTS idx_audit_priority ON audit_assignments(run_id, audit_stratum, priority_order);
