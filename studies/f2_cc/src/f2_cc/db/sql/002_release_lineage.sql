CREATE TABLE releases (
    release_id TEXT PRIMARY KEY, profile_key TEXT NOT NULL, source_run_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs,
    manifest_uri TEXT NOT NULL UNIQUE, manifest_sha256 VARCHAR(64) NOT NULL, status TEXT NOT NULL,
    statistics JSONB NOT NULL DEFAULT '{}'::jsonb, published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE release_artifacts (
    release_id TEXT NOT NULL REFERENCES releases ON DELETE RESTRICT, role TEXT NOT NULL, uri TEXT NOT NULL UNIQUE,
    sha256 VARCHAR(64) NOT NULL, byte_size BIGINT NOT NULL, record_count BIGINT, format TEXT NOT NULL,
    PRIMARY KEY(release_id,role,uri)
);
CREATE TABLE stage_lineage (
    stage_id TEXT PRIMARY KEY, source_run_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs,
    stage_role TEXT NOT NULL, config_hash VARCHAR(64) NOT NULL, evidence_digest VARCHAR(64) NOT NULL,
    status TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(source_run_id,stage_role,config_hash,evidence_digest)
);
