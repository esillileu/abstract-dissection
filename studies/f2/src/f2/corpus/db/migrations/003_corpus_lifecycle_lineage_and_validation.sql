-- Migration 003: Generic Corpus Lifecycle, Multi-Hop Lineage, and Validation Evidence

-- 1. Acquisition Runs (References catalog.resource_versions as SSOT for raw release)
CREATE TABLE IF NOT EXISTS acquisition_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    resource_version_id TEXT NOT NULL REFERENCES catalog.resource_versions(resource_version_id) ON DELETE RESTRICT,
    acquisition_method VARCHAR(64) NOT NULL, -- 'http_archive_download', 'cdx_range_fetch', 'git_clone', 'manual_import'
    code_version VARCHAR(64) NOT NULL,       -- Git commit hash
    config_hash VARCHAR(64) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(32) NOT NULL DEFAULT 'running', -- 'running', 'completed', 'failed'
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    log_s3_uri TEXT,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_acq_runs_version ON acquisition_runs(resource_version_id);

-- 2. Immutable Artifacts (Every object stored in SeaweedFS S3)
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id VARCHAR(64) PRIMARY KEY,
    stage VARCHAR(32) NOT NULL, -- 'raw', 'extracted', 'normalized', 'filtered', 'canonical_shard', 'report'
    s3_uri TEXT NOT NULL UNIQUE,
    sha256 VARCHAR(64) NOT NULL,
    byte_size BIGINT NOT NULL,
    format VARCHAR(32) NOT NULL, -- 'warc.gz', 'arc.gz', 'tar.gz', 'txt', 'parquet', 'jsonl.gz'
    record_count BIGINT,
    resource_version_id TEXT REFERENCES catalog.resource_versions(resource_version_id) ON DELETE RESTRICT,
    acquisition_run_id VARCHAR(64) REFERENCES acquisition_runs(run_id) ON DELETE RESTRICT,
    integrity_status VARCHAR(32) NOT NULL DEFAULT 'pending', -- 'pending', 'verified', 'corrupt'
    verified_at TIMESTAMPTZ,
    verification_report JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_artifacts_stage ON artifacts(stage);
CREATE INDEX IF NOT EXISTS idx_artifacts_sha256 ON artifacts(sha256);
CREATE INDEX IF NOT EXISTS idx_artifacts_resource_version ON artifacts(resource_version_id);

-- 3. Processing Runs & N:M Multi-Hop DAG Junctions (RESTRICT on deletion)
CREATE TABLE IF NOT EXISTS processing_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    recipe_name VARCHAR(64) NOT NULL,    -- 'warc_extractor', 'text_normalizer', 'news_dedup_filter', 'word2vec_sharder'
    recipe_version VARCHAR(32) NOT NULL, -- Semantic version
    code_version VARCHAR(64) NOT NULL,   -- Git commit hash
    config_hash VARCHAR(64) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(32) NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    log_s3_uri TEXT,
    diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS processing_run_inputs (
    run_id VARCHAR(64) NOT NULL REFERENCES processing_runs(run_id) ON DELETE RESTRICT,
    artifact_id VARCHAR(64) NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
    input_role VARCHAR(32) NOT NULL DEFAULT 'primary', -- 'primary', 'rules', 'dictionary'
    PRIMARY KEY (run_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS processing_run_outputs (
    run_id VARCHAR(64) NOT NULL REFERENCES processing_runs(run_id) ON DELETE RESTRICT,
    artifact_id VARCHAR(64) NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
    output_role VARCHAR(32) NOT NULL DEFAULT 'primary', -- 'clean_shard', 'reject_log', 'diagnostics'
    PRIMARY KEY (run_id, artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_proc_inputs_artifact ON processing_run_inputs(artifact_id);
CREATE INDEX IF NOT EXISTS idx_proc_outputs_artifact ON processing_run_outputs(artifact_id);

-- 4. Corpus Version Shards & Summary Stats (RESTRICT on deletion)
CREATE TABLE IF NOT EXISTS corpus_shards (
    resource_version_id TEXT NOT NULL REFERENCES catalog.resource_versions(resource_version_id) ON DELETE RESTRICT,
    shard_index INT NOT NULL,
    artifact_id VARCHAR(64) NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
    word_count BIGINT NOT NULL DEFAULT 0,
    doc_count BIGINT NOT NULL DEFAULT 0,
    byte_size BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (resource_version_id, shard_index),
    CONSTRAINT uq_corpus_shard_artifact UNIQUE (resource_version_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS corpus_version_stats (
    resource_version_id TEXT PRIMARY KEY REFERENCES catalog.resource_versions(resource_version_id) ON DELETE RESTRICT,
    total_words BIGINT NOT NULL DEFAULT 0,
    total_tokens BIGINT NOT NULL DEFAULT 0,
    total_documents BIGINT NOT NULL DEFAULT 0,
    total_sentences BIGINT NOT NULL DEFAULT 0,
    total_bytes BIGINT NOT NULL DEFAULT 0,
    total_shards INT NOT NULL DEFAULT 0,
    source_composition JSONB NOT NULL DEFAULT '{}'::jsonb,
    year_distribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Immutable Versioned Validation Profiles
CREATE TABLE IF NOT EXISTS validation_profiles (
    profile_id VARCHAR(64) PRIMARY KEY,       -- e.g. 'word2vec_news_eval_r1'
    profile_key VARCHAR(64) NOT NULL,         -- e.g. 'word2vec_news_eval'
    revision INT NOT NULL,                    -- 1, 2, 3...
    name TEXT NOT NULL,
    description TEXT,
    target_paper_id TEXT REFERENCES catalog.papers(paper_id) ON DELETE RESTRICT,
    spec_hash VARCHAR(64) NOT NULL,           -- SHA-256 of specification JSON
    specification JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    superseded_at TIMESTAMPTZ,
    notes TEXT,
    CONSTRAINT uq_validation_profile_revision UNIQUE (profile_key, revision)
);

CREATE OR REPLACE FUNCTION prevent_validation_profile_modification()
RETURNS TRIGGER AS $$
BEGIN
    IF (OLD.spec_hash IS DISTINCT FROM NEW.spec_hash OR
        OLD.specification IS DISTINCT FROM NEW.specification OR
        OLD.profile_key IS DISTINCT FROM NEW.profile_key OR
        OLD.revision IS DISTINCT FROM NEW.revision OR
        OLD.target_paper_id IS DISTINCT FROM NEW.target_paper_id) THEN
        RAISE EXCEPTION 'Validation profile specification is immutable: profile_id=%, revision=%. Target paper and specifications cannot be modified. Create a new revision instead.', OLD.profile_id, OLD.revision;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_prevent_validation_profile_modification ON validation_profiles;
CREATE TRIGGER trg_prevent_validation_profile_modification
BEFORE UPDATE ON validation_profiles
FOR EACH ROW
EXECUTE FUNCTION prevent_validation_profile_modification();

-- 6. Validation Runs & Multi-Target Referential Integrity
CREATE TABLE IF NOT EXISTS validation_runs (
    validation_run_id VARCHAR(64) PRIMARY KEY,
    profile_id VARCHAR(64) NOT NULL REFERENCES validation_profiles(profile_id) ON DELETE RESTRICT,
    
    -- Target definition with DB-enforced referential integrity
    target_type VARCHAR(32) NOT NULL, -- 'processing_run', 'resource_version', 'artifact'
    target_processing_run_id VARCHAR(64) REFERENCES processing_runs(run_id) ON DELETE RESTRICT,
    target_artifact_id VARCHAR(64) REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
    target_resource_version_id TEXT REFERENCES catalog.resource_versions(resource_version_id) ON DELETE RESTRICT,
    target_id TEXT GENERATED ALWAYS AS (COALESCE(target_processing_run_id::text, target_artifact_id::text, target_resource_version_id)) STORED,
    
    validator_code_version VARCHAR(64) NOT NULL, -- Git commit of validation engine
    validator_config_hash VARCHAR(64) NOT NULL,
    validator_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    overall_verdict VARCHAR(16) NOT NULL,     -- 'PASS', 'WARN', 'FAIL'
    paper_compatibility VARCHAR(32) NOT NULL DEFAULT 'not_applicable', -- 'exact', 'compatible_reconstruction', 'incompatible'
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    evidence_report_uri TEXT,
    summary_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT,

    CONSTRAINT chk_validation_run_target CHECK (
        (target_type = 'processing_run' AND target_processing_run_id IS NOT NULL AND target_artifact_id IS NULL AND target_resource_version_id IS NULL) OR
        (target_type = 'artifact' AND target_artifact_id IS NOT NULL AND target_processing_run_id IS NULL AND target_resource_version_id IS NULL) OR
        (target_type = 'resource_version' AND target_resource_version_id IS NOT NULL AND target_processing_run_id IS NULL AND target_artifact_id IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS validation_checks (
    check_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    validation_run_id VARCHAR(64) NOT NULL REFERENCES validation_runs(validation_run_id) ON DELETE CASCADE,
    check_name VARCHAR(64) NOT NULL,
    category VARCHAR(32) NOT NULL,            -- 'integrity', 'transformation', 'structural', 'statistical', 'compatibility'
    status VARCHAR(16) NOT NULL,              -- 'PASS', 'WARN', 'FAIL'
    expected_condition TEXT NOT NULL,
    observed_value TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_validation_runs_target ON validation_runs(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_validation_runs_profile ON validation_runs(profile_id);
CREATE INDEX IF NOT EXISTS idx_validation_checks_run ON validation_checks(validation_run_id);

-- 7. Common Crawl Bridge Relation (Preserving existing pipeline_runs table untouched, with PK & retry uniqueness)
CREATE TABLE IF NOT EXISTS pipeline_run_lineage (
    lineage_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_run_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE RESTRICT,
    acquisition_run_id VARCHAR(64) REFERENCES acquisition_runs(run_id) ON DELETE RESTRICT,
    processing_run_id VARCHAR(64) REFERENCES processing_runs(run_id) ON DELETE RESTRICT,
    stage_role VARCHAR(32) NOT NULL, -- 'cdx_discovery', 'arc_range_fetch', 'html_extract', 'reject_explore', 'shard_text'
    execution_attempt INT NOT NULL DEFAULT 1,
    linked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    CONSTRAINT chk_pipeline_lineage_target CHECK (
        (acquisition_run_id IS NOT NULL AND processing_run_id IS NULL) OR
        (acquisition_run_id IS NULL AND processing_run_id IS NOT NULL)
    ),
    CONSTRAINT chk_pipeline_lineage_attempt CHECK (execution_attempt >= 1)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pipeline_lineage_acq
    ON pipeline_run_lineage (pipeline_run_id, stage_role, execution_attempt, acquisition_run_id)
    WHERE acquisition_run_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_pipeline_lineage_proc
    ON pipeline_run_lineage (pipeline_run_id, stage_role, execution_attempt, processing_run_id)
    WHERE processing_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pipeline_lineage_run ON pipeline_run_lineage(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_lineage_acq ON pipeline_run_lineage(acquisition_run_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_lineage_proc ON pipeline_run_lineage(processing_run_id);
