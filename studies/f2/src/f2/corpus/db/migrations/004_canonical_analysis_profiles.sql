-- Immutable canonical analysis/calibration run selection.

CREATE TABLE IF NOT EXISTS analysis_profiles (
    profile_key VARCHAR(64) NOT NULL,
    revision INT NOT NULL,
    purpose VARCHAR(32) NOT NULL,
    run_id VARCHAR(64) NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE RESTRICT,
    rationale TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (profile_key, revision),
    CONSTRAINT uq_analysis_profile_revision_run UNIQUE (profile_key, revision, run_id),
    CONSTRAINT chk_analysis_profile_purpose CHECK (purpose IN ('analysis', 'calibration'))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_active_analysis_profile
    ON analysis_profiles(profile_key) WHERE is_active;

CREATE OR REPLACE FUNCTION prevent_analysis_profile_immutable_fields()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.profile_key IS DISTINCT FROM NEW.profile_key
       OR OLD.revision IS DISTINCT FROM NEW.revision
       OR OLD.purpose IS DISTINCT FROM NEW.purpose
       OR OLD.run_id IS DISTINCT FROM NEW.run_id
       OR OLD.rationale IS DISTINCT FROM NEW.rationale
       OR OLD.created_at IS DISTINCT FROM NEW.created_at THEN
        RAISE EXCEPTION 'Analysis profile mapping is immutable; create a new revision';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_analysis_profile_immutable ON analysis_profiles;
CREATE TRIGGER trg_analysis_profile_immutable
BEFORE UPDATE ON analysis_profiles
FOR EACH ROW EXECUTE FUNCTION prevent_analysis_profile_immutable_fields();

-- The migration is safe to apply before the historical runs are restored.
-- The bootstrap command can call the same inserts once those runs exist.
INSERT INTO analysis_profiles (profile_key, revision, purpose, run_id, rationale)
SELECT v.profile_key, 1, v.purpose, v.run_id, v.rationale
FROM (VALUES
    ('confirmatory-50k', 'analysis', 'run_50k_confirmatory', 'Canonical confirmatory 50K run approved by F2 integration plan'),
    ('calibration-10k', 'calibration', 'run_42_a1d3745e', 'Canonical calibration 10K run approved by F2 integration plan')
) AS v(profile_key, purpose, run_id, rationale)
WHERE EXISTS (SELECT 1 FROM pipeline_runs p WHERE p.run_id = v.run_id)
ON CONFLICT (profile_key, revision) DO NOTHING;
