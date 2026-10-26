-- Experiment completion belongs only to MLflow. Preserve historical SQL ledgers.
DROP VIEW v_canonical_plan_progress;
DROP VIEW v_canonical_run_matrix;
ALTER TABLE planned_run_slots DROP COLUMN reference_mlflow_run_id;

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
    prs.created_at;

-- View 2: v_canonical_plan_inventory
-- Expected experiment slots for the active canonical plan
CREATE OR REPLACE VIEW v_canonical_plan_inventory AS
SELECT
    ep.plan_key,
    ep.revision AS plan_revision,
    ep.execution_plan_id,
    epe.plan_experiment_id,
    es.experiment_spec_id,
    es.name AS experiment_name,
    COUNT(prs.planned_run_slot_id) FILTER (WHERE prs.expected) AS expected_slots
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

