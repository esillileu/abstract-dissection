-- Enforce the two canonical run/profile mappings without creating data.

CREATE OR REPLACE FUNCTION assert_canonical_analysis_profile_contract()
RETURNS VOID AS $$
DECLARE
    run_present BOOLEAN;
    mapping_count BIGINT;
BEGIN
    SELECT EXISTS (SELECT 1 FROM pipeline_runs WHERE run_id = 'run_50k_confirmatory') INTO run_present;
    SELECT COUNT(*) FROM analysis_profiles
      WHERE profile_key = 'confirmatory-50k' AND run_id = 'run_50k_confirmatory' AND is_active
      INTO mapping_count;
    IF run_present IS DISTINCT FROM (mapping_count = 1) THEN
        RAISE EXCEPTION 'canonical contract mismatch: confirmatory-50k must map exactly once to run_50k_confirmatory';
    END IF;

    SELECT EXISTS (SELECT 1 FROM pipeline_runs WHERE run_id = 'run_42_a1d3745e') INTO run_present;
    SELECT COUNT(*) FROM analysis_profiles
      WHERE profile_key = 'calibration-10k' AND run_id = 'run_42_a1d3745e' AND is_active
      INTO mapping_count;
    IF run_present IS DISTINCT FROM (mapping_count = 1) THEN
        RAISE EXCEPTION 'canonical contract mismatch: calibration-10k must map exactly once to run_42_a1d3745e';
    END IF;

    IF EXISTS (
        SELECT 1 FROM analysis_profiles
        WHERE is_active AND (
          (profile_key = 'confirmatory-50k' AND run_id <> 'run_50k_confirmatory') OR
          (profile_key = 'calibration-10k' AND run_id <> 'run_42_a1d3745e')
        )
    ) THEN
        RAISE EXCEPTION 'canonical profile maps to the wrong run';
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION enforce_canonical_analysis_profile_contract()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM assert_canonical_analysis_profile_contract();
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_canonical_profile_contract_profiles ON analysis_profiles;
CREATE CONSTRAINT TRIGGER trg_canonical_profile_contract_profiles
AFTER INSERT OR UPDATE OR DELETE ON analysis_profiles
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION enforce_canonical_analysis_profile_contract();

DROP TRIGGER IF EXISTS trg_canonical_profile_contract_runs ON pipeline_runs;
CREATE CONSTRAINT TRIGGER trg_canonical_profile_contract_runs
AFTER INSERT OR UPDATE OR DELETE ON pipeline_runs
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION enforce_canonical_analysis_profile_contract();

-- Reject an already-inconsistent database during migration.
SELECT assert_canonical_analysis_profile_contract();
