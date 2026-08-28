-- Migration 001: Initial F2 Reproduction Catalog Database Schema

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(64) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 1. Papers
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    citation_key TEXT,
    version TEXT,
    source_url TEXT,
    notes TEXT
);

-- 2. Reproduction Targets
CREATE TABLE IF NOT EXISTS reproduction_targets (
    target_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    location_type TEXT NOT NULL,
    location_label TEXT NOT NULL,
    ordinal INT,
    target_type TEXT NOT NULL,
    description TEXT NOT NULL,
    source_locator TEXT,
    notes TEXT
);

-- 3. Experiment Specs
CREATE TABLE IF NOT EXISTS experiment_specs (
    experiment_spec_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    run_type TEXT NOT NULL,
    provenance_status TEXT NOT NULL,
    specification JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_locator TEXT,
    notes TEXT
);

-- 4. Target Experiments Link Table
CREATE TABLE IF NOT EXISTS target_experiments (
    target_id TEXT NOT NULL REFERENCES reproduction_targets(target_id) ON DELETE CASCADE,
    experiment_spec_id TEXT NOT NULL REFERENCES experiment_specs(experiment_spec_id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'primary',
    PRIMARY KEY (target_id, experiment_spec_id)
);

-- 5. Resources
CREATE TABLE IF NOT EXISTS resources (
    resource_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    access_status TEXT NOT NULL,
    acquisition_status TEXT NOT NULL,
    readiness_status TEXT NOT NULL,
    canonical_version_id TEXT,
    notes TEXT
);

-- 6. Resource Versions
CREATE TABLE IF NOT EXISTS resource_versions (
    resource_version_id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL REFERENCES resources(resource_id) ON DELETE CASCADE,
    version_label TEXT,
    uri TEXT,
    local_path TEXT,
    checksum_algo TEXT,
    checksum TEXT,
    size_bytes BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    CONSTRAINT uq_resource_version_composite UNIQUE (resource_id, resource_version_id)
);

-- Invariant: Canonical version must belong to the exact resource
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_resources_canonical_version'
    ) THEN
        ALTER TABLE resources
            ADD CONSTRAINT fk_resources_canonical_version
            FOREIGN KEY (resource_id, canonical_version_id)
            REFERENCES resource_versions(resource_id, resource_version_id)
            ON DELETE SET NULL;
    END IF;
END $$;

-- 7. Resource Sources
CREATE TABLE IF NOT EXISTS resource_sources (
    source_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    resource_id TEXT NOT NULL REFERENCES resources(resource_id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    url TEXT,
    citation TEXT,
    license TEXT,
    is_preferred BOOLEAN NOT NULL DEFAULT FALSE,
    notes TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_resource_preferred_source
    ON resource_sources(resource_id)
    WHERE is_preferred;

-- 8. Preparation Specs
CREATE TABLE IF NOT EXISTS preparation_specs (
    preparation_id TEXT PRIMARY KEY,
    input_resource_id TEXT NOT NULL REFERENCES resources(resource_id) ON DELETE CASCADE,
    output_resource_id TEXT REFERENCES resources(resource_id) ON DELETE SET NULL,
    preparation_type TEXT NOT NULL,
    specification JSONB NOT NULL DEFAULT '{}'::jsonb,
    code_resource_id TEXT REFERENCES resources(resource_id) ON DELETE SET NULL,
    status TEXT NOT NULL,
    notes TEXT
);

-- 9. Experiment Requirements
CREATE TABLE IF NOT EXISTS experiment_requirements (
    requirement_id TEXT PRIMARY KEY,
    experiment_spec_id TEXT NOT NULL REFERENCES experiment_specs(experiment_spec_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    required_resource_id TEXT REFERENCES resources(resource_id) ON DELETE SET NULL,
    requirement_spec JSONB,
    required BOOLEAN NOT NULL DEFAULT TRUE,
    notes TEXT,
    CONSTRAINT uq_exp_req_composite UNIQUE (experiment_spec_id, requirement_id)
);

-- 10. Requirement Candidates
CREATE TABLE IF NOT EXISTS requirement_candidates (
    requirement_id TEXT NOT NULL REFERENCES experiment_requirements(requirement_id) ON DELETE CASCADE,
    resource_id TEXT NOT NULL REFERENCES resources(resource_id) ON DELETE CASCADE,
    candidate_type TEXT NOT NULL,
    status TEXT NOT NULL,
    justification TEXT,
    notes TEXT,
    PRIMARY KEY (requirement_id, resource_id)
);

-- 11. Execution Plans
CREATE TABLE IF NOT EXISTS execution_plans (
    execution_plan_id TEXT PRIMARY KEY,
    plan_key TEXT NOT NULL,
    revision INT NOT NULL,
    status TEXT NOT NULL,
    source_ref TEXT,
    source_hash TEXT,
    is_canonical BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    superseded_at TIMESTAMPTZ,
    notes TEXT,
    CONSTRAINT uq_execution_plan_key_rev UNIQUE (plan_key, revision)
);

-- Invariant: At most one canonical plan per plan_key
CREATE UNIQUE INDEX IF NOT EXISTS uq_canonical_plan_key
    ON execution_plans(plan_key)
    WHERE is_canonical;

-- 12. Execution Plan Experiments
CREATE TABLE IF NOT EXISTS execution_plan_experiments (
    plan_experiment_id TEXT PRIMARY KEY,
    execution_plan_id TEXT NOT NULL REFERENCES execution_plans(execution_plan_id) ON DELETE CASCADE,
    experiment_spec_id TEXT NOT NULL REFERENCES experiment_specs(experiment_spec_id) ON DELETE CASCADE,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT,
    CONSTRAINT uq_execution_plan_spec UNIQUE (execution_plan_id, experiment_spec_id),
    CONSTRAINT uq_plan_exp_spec_composite UNIQUE (plan_experiment_id, experiment_spec_id)
);

-- 13. Execution Plan Bindings
CREATE TABLE IF NOT EXISTS execution_plan_bindings (
    plan_experiment_id TEXT NOT NULL REFERENCES execution_plan_experiments(plan_experiment_id) ON DELETE CASCADE,
    requirement_id TEXT NOT NULL REFERENCES experiment_requirements(requirement_id) ON DELETE CASCADE,
    resource_version_id TEXT NOT NULL REFERENCES resource_versions(resource_version_id) ON DELETE RESTRICT,
    binding_type TEXT NOT NULL,
    justification TEXT,
    notes TEXT,
    PRIMARY KEY (plan_experiment_id, requirement_id)
);

-- Invariant Trigger: Plan binding requirement must belong to the exact experiment_spec of plan_experiment
CREATE OR REPLACE FUNCTION check_plan_binding_integrity()
RETURNS TRIGGER AS $$
DECLARE
    v_plan_spec_id TEXT;
    v_req_spec_id TEXT;
BEGIN
    SELECT experiment_spec_id INTO v_plan_spec_id
    FROM execution_plan_experiments
    WHERE plan_experiment_id = NEW.plan_experiment_id;

    SELECT experiment_spec_id INTO v_req_spec_id
    FROM experiment_requirements
    WHERE requirement_id = NEW.requirement_id;

    IF v_plan_spec_id IS NULL THEN
        RAISE EXCEPTION 'Plan experiment % does not exist.', NEW.plan_experiment_id;
    END IF;
    IF v_req_spec_id IS NULL THEN
        RAISE EXCEPTION 'Experiment requirement % does not exist.', NEW.requirement_id;
    END IF;
    IF v_plan_spec_id <> v_req_spec_id THEN
        RAISE EXCEPTION 'Plan binding integrity violation: requirement % belongs to experiment_spec %, but plan_experiment % belongs to experiment_spec %.',
            NEW.requirement_id, v_req_spec_id, NEW.plan_experiment_id, v_plan_spec_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_check_plan_binding_integrity ON execution_plan_bindings;
CREATE TRIGGER trg_check_plan_binding_integrity
    BEFORE INSERT OR UPDATE ON execution_plan_bindings
    FOR EACH ROW
    EXECUTE FUNCTION check_plan_binding_integrity();

-- 14. Planned Run Slots
CREATE TABLE IF NOT EXISTS planned_run_slots (
    planned_run_slot_id TEXT PRIMARY KEY,
    plan_experiment_id TEXT NOT NULL REFERENCES execution_plan_experiments(plan_experiment_id) ON DELETE CASCADE,
    slot_key TEXT NOT NULL,
    atomic_run_id TEXT,
    variant_key TEXT,
    seed BIGINT,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    expected BOOLEAN NOT NULL DEFAULT TRUE,
    reference_mlflow_run_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    CONSTRAINT uq_planned_run_slot UNIQUE (plan_experiment_id, slot_key)
);

-- 15. Reported Results
CREATE TABLE IF NOT EXISTS reported_results (
    reported_result_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    target_id TEXT NOT NULL REFERENCES reproduction_targets(target_id) ON DELETE CASCADE,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION,
    value_text TEXT,
    unit TEXT,
    aggregation TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_targets_paper ON reproduction_targets(paper_id);
CREATE INDEX IF NOT EXISTS idx_exp_specs_paper ON experiment_specs(paper_id);
CREATE INDEX IF NOT EXISTS idx_resources_kind ON resources(kind);
CREATE INDEX IF NOT EXISTS idx_resource_versions_res ON resource_versions(resource_id);
CREATE INDEX IF NOT EXISTS idx_resource_sources_res ON resource_sources(resource_id);
CREATE INDEX IF NOT EXISTS idx_requirements_spec ON experiment_requirements(experiment_spec_id);
CREATE INDEX IF NOT EXISTS idx_req_candidates_req ON requirement_candidates(requirement_id);
CREATE INDEX IF NOT EXISTS idx_req_candidates_res ON requirement_candidates(resource_id);
CREATE INDEX IF NOT EXISTS idx_plan_exps_plan ON execution_plan_experiments(execution_plan_id);
CREATE INDEX IF NOT EXISTS idx_plan_bindings_plan_exp ON execution_plan_bindings(plan_experiment_id);
CREATE INDEX IF NOT EXISTS idx_plan_bindings_version ON execution_plan_bindings(resource_version_id);
CREATE INDEX IF NOT EXISTS idx_planned_slots_exp ON planned_run_slots(plan_experiment_id);
CREATE INDEX IF NOT EXISTS idx_planned_slots_mlflow ON planned_run_slots(reference_mlflow_run_id) WHERE reference_mlflow_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_reported_results_target ON reported_results(target_id);

-- Views

-- View 1: v_canonical_run_matrix
-- Expected run slots for the active canonical plan
CREATE OR REPLACE VIEW v_canonical_run_matrix AS
SELECT
    ep.plan_key,
    ep.revision AS plan_revision,
    ep.execution_plan_id,
    epe.plan_experiment_id,
    es.experiment_spec_id,
    es.name AS experiment_name,
    COALESCE(
        ARRAY_AGG(DISTINCT te.target_id) FILTER (WHERE te.target_id IS NOT NULL),
        ARRAY[]::text[]
    ) AS target_ids,
    prs.planned_run_slot_id,
    prs.slot_key,
    prs.atomic_run_id,
    prs.variant_key,
    prs.seed,
    prs.parameters,
    prs.expected,
    prs.reference_mlflow_run_id,
    (prs.reference_mlflow_run_id IS NOT NULL) AS executed,
    prs.created_at AS slot_created_at
FROM execution_plans ep
JOIN execution_plan_experiments epe
    ON epe.execution_plan_id = ep.execution_plan_id
JOIN experiment_specs es
    ON es.experiment_spec_id = epe.experiment_spec_id
JOIN planned_run_slots prs
    ON prs.plan_experiment_id = epe.plan_experiment_id
LEFT JOIN target_experiments te
    ON te.experiment_spec_id = es.experiment_spec_id
WHERE ep.is_canonical = TRUE
GROUP BY
    ep.plan_key,
    ep.revision,
    ep.execution_plan_id,
    epe.plan_experiment_id,
    es.experiment_spec_id,
    es.name,
    prs.planned_run_slot_id,
    prs.slot_key,
    prs.atomic_run_id,
    prs.variant_key,
    prs.seed,
    prs.parameters,
    prs.expected,
    prs.reference_mlflow_run_id,
    prs.created_at;

-- View 2: v_canonical_plan_progress
-- Experiment-level and overall progress for active canonical plan
CREATE OR REPLACE VIEW v_canonical_plan_progress AS
SELECT
    ep.plan_key,
    ep.revision AS plan_revision,
    ep.execution_plan_id,
    epe.plan_experiment_id,
    es.experiment_spec_id,
    es.name AS experiment_name,
    COUNT(prs.planned_run_slot_id) FILTER (WHERE prs.expected) AS expected_slots,
    COUNT(prs.planned_run_slot_id) FILTER (WHERE prs.expected AND prs.reference_mlflow_run_id IS NOT NULL) AS executed_slots,
    COUNT(prs.planned_run_slot_id) FILTER (WHERE prs.expected AND prs.reference_mlflow_run_id IS NULL) AS missing_slots,
    CASE
        WHEN COUNT(prs.planned_run_slot_id) FILTER (WHERE prs.expected) = 0 THEN 0.0
        ELSE ROUND(
            (COUNT(prs.planned_run_slot_id) FILTER (WHERE prs.expected AND prs.reference_mlflow_run_id IS NOT NULL)::numeric /
             COUNT(prs.planned_run_slot_id) FILTER (WHERE prs.expected)::numeric),
            4
        )::double precision
    END AS completion_rate
FROM execution_plans ep
JOIN execution_plan_experiments epe
    ON epe.execution_plan_id = ep.execution_plan_id
JOIN experiment_specs es
    ON es.experiment_spec_id = epe.experiment_spec_id
LEFT JOIN planned_run_slots prs
    ON prs.plan_experiment_id = epe.plan_experiment_id
WHERE ep.is_canonical = TRUE
GROUP BY
    ep.plan_key,
    ep.revision,
    ep.execution_plan_id,
    epe.plan_experiment_id,
    es.experiment_spec_id,
    es.name;

-- View 3: v_resource_inventory
-- Resource status, preferred sources, and dependency blockers
CREATE OR REPLACE VIEW v_resource_inventory AS
SELECT
    r.resource_id,
    r.kind,
    r.name,
    r.description,
    r.access_status,
    r.acquisition_status,
    r.readiness_status,
    r.canonical_version_id,
    pref_s.source_type AS preferred_source_type,
    pref_s.url AS preferred_source_url,
    pref_s.license AS preferred_source_license,
    COUNT(DISTINCT er.requirement_id) AS dependent_requirement_count,
    COUNT(DISTINCT rc.requirement_id) AS candidate_for_requirement_count,
    COUNT(DISTINCT rv.resource_version_id) AS version_count
FROM resources r
LEFT JOIN resource_sources pref_s
    ON pref_s.resource_id = r.resource_id AND pref_s.is_preferred = TRUE
LEFT JOIN experiment_requirements er
    ON er.required_resource_id = r.resource_id
LEFT JOIN requirement_candidates rc
    ON rc.resource_id = r.resource_id
LEFT JOIN resource_versions rv
    ON rv.resource_id = r.resource_id
GROUP BY
    r.resource_id,
    r.kind,
    r.name,
    r.description,
    r.access_status,
    r.acquisition_status,
    r.readiness_status,
    r.canonical_version_id,
    pref_s.source_type,
    pref_s.url,
    pref_s.license;
