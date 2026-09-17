-- Common Crawl operational ownership moved to the physically separate f2_cc DB.
-- This migration runs only after the cross-database fingerprint-verified transfer.

DROP TRIGGER IF EXISTS trg_canonical_profile_contract_profiles ON analysis_profiles;
DROP TRIGGER IF EXISTS trg_canonical_profile_contract_runs ON pipeline_runs;
DROP FUNCTION IF EXISTS enforce_canonical_analysis_profile_contract();
DROP FUNCTION IF EXISTS assert_canonical_analysis_profile_contract();

DROP TABLE IF EXISTS pipeline_run_lineage;
DROP TABLE IF EXISTS analysis_profiles;
DROP TABLE IF EXISTS audit_assignments;
DROP TABLE IF EXISTS processing_results;
DROP TABLE IF EXISTS candidate_records;
DROP TABLE IF EXISTS pipeline_runs;
