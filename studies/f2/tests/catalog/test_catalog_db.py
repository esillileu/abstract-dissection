"""Integration tests for PostgreSQL F2 Reproduction Catalog DB schema, invariants, and views."""

from __future__ import annotations

import os
import uuid
from typing import Any

import psycopg
import pytest

from f2.catalog.db.migrations.runner import run_catalog_migrations
from f2.catalog.db.repository import CatalogRepository
from f2.catalog.db.session import get_catalog_db_url, get_connection
from f2.catalog.materializer import CatalogPlanMaterializer
from repro_core.execution.definition import RunPlan


@pytest.fixture
def catalog_db_conn():
    url = os.getenv("F2_CATALOG_DATABASE_URL") or get_catalog_db_url()
    try:
        with get_connection(url) as conn:
            run_catalog_migrations(conn)
            yield conn
    except Exception as exc:
        pytest.skip(f"F2 Catalog DB connection not available: {exc}")


@pytest.fixture
def repo(catalog_db_conn: psycopg.Connection[Any]):
    # Run tests inside an isolated transaction that rolls back automatically
    with catalog_db_conn.transaction(force_rollback=True):
        yield CatalogRepository(catalog_db_conn)


def test_catalog_migrations_and_idempotency(catalog_db_conn: psycopg.Connection[Any]):
    """Ensure running catalog migrations again is a clean, idempotent no-op."""
    applied = run_catalog_migrations(catalog_db_conn)
    assert applied == []

    with catalog_db_conn.cursor() as cur:
        cur.execute(
            "SELECT version FROM schema_migrations WHERE version = '001_initial_catalog_schema';"
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "001_initial_catalog_schema"


def test_canonical_plan_revision_lifecycle_and_uniqueness(repo: CatalogRepository):
    """Test canonical plan revision transitions and uniqueness invariant."""
    test_key = f"plan_test_{uuid.uuid4().hex[:6]}"

    # 1. Create rev 1 as canonical
    repo.upsert_execution_plan(
        execution_plan_id=f"{test_key}_r1",
        plan_key=test_key,
        revision=1,
        status="runnable",
        is_canonical=True,
        notes="Initial canonical plan",
    )

    # 2. Create rev 2 as draft
    repo.upsert_execution_plan(
        execution_plan_id=f"{test_key}_r2",
        plan_key=test_key,
        revision=2,
        status="draft",
        is_canonical=False,
        notes="Second revision draft",
    )

    # Verify rev1 is canonical and rev2 is not
    with repo.conn.cursor() as cur:
        cur.execute(
            "SELECT execution_plan_id, revision, is_canonical, superseded_at FROM execution_plans WHERE plan_key = %s ORDER BY revision;",
            (test_key,),
        )
        plans = cur.fetchall()
        assert len(plans) == 2
        assert plans[0][1] == 1 and plans[0][2] is True and plans[0][3] is None
        assert plans[1][1] == 2 and plans[1][2] is False

    # 3. Promote rev 2 to canonical
    repo.set_canonical_execution_plan(
        execution_plan_id=f"{test_key}_r2", plan_key=test_key
    )

    # Verify rev1 is demoted and rev2 is canonical
    with repo.conn.cursor() as cur:
        cur.execute(
            "SELECT execution_plan_id, revision, is_canonical, superseded_at FROM execution_plans WHERE plan_key = %s ORDER BY revision;",
            (test_key,),
        )
        plans = cur.fetchall()
        # Rev 1 is preserved with is_canonical = False and superseded_at timestamp populated
        assert plans[0][1] == 1 and plans[0][2] is False and plans[0][3] is not None
        # Rev 2 is canonical
        assert plans[1][1] == 2 and plans[1][2] is True and plans[1][3] is None

    # 4. Invariant: Attempting to insert a second canonical plan directly must violate partial unique index
    with pytest.raises(psycopg.errors.UniqueViolation):
        with repo.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO execution_plans (execution_plan_id, plan_key, revision, status, is_canonical)
                VALUES (%s, %s, 3, 'runnable', TRUE);
                """,
                (f"{test_key}_r3", test_key),
            )


def test_planned_run_coverage_and_progress_view(repo: CatalogRepository):
    """Test materializing 2 variants x 3 seeds (6 slots), executing 4, and checking missing seeds."""
    test_key = f"cov_test_{uuid.uuid4().hex[:6]}"
    paper_id = f"paper_{uuid.uuid4().hex[:6]}"
    spec_id = f"{test_key}_e01"
    target_id = f"target_{uuid.uuid4().hex[:6]}"

    # Setup paper, target, and experiment spec
    repo.upsert_paper(paper_id=paper_id, title="Test Word2Vec Paper")
    repo.upsert_target(
        target_id=target_id,
        paper_id=paper_id,
        location_type="table",
        location_label="Table 1",
        target_type="accuracy",
        description="Semantic Accuracy",
    )
    repo.upsert_experiment_spec(
        experiment_spec_id=spec_id,
        paper_id=paper_id,
        name="Word2Vec Baseline",
        run_type="trained",
        provenance_status="explicit",
    )
    repo.link_target_experiment(target_id=target_id, experiment_spec_id=spec_id)

    # Create synthetic RunPlans: 2 variants ('cbow', 'skipgram') x 3 seeds (42, 43, 44) = 6 slots
    run_plans: list[RunPlan] = []
    for variant in ["cbow", "skipgram"]:
        for seed in [42, 43, 44]:
            run_plans.append(
                RunPlan(
                    domain=test_key,
                    experiment_id="e01",
                    path=os.path.abspath(__file__),
                    atomic_run_id=variant,
                    seed=seed,
                    device="cpu",
                )
            )

    materializer = CatalogPlanMaterializer(repo)
    exec_plan_id = materializer.materialize_from_plans(
        plan_key=test_key,
        revision=1,
        plans=run_plans,
        is_canonical=True,
    )
    assert exec_plan_id == f"{test_key}_r1"

    # Query v_canonical_plan_progress initially: 6 expected, 0 executed, 6 missing
    progress = repo.get_canonical_plan_progress(plan_key=test_key)
    assert len(progress) == 1
    p = progress[0]
    assert p["expected_slots"] == 6
    assert p["executed_slots"] == 0
    assert p["missing_slots"] == 6
    assert p["completion_rate"] == 0.0

    # Query initial run matrix
    matrix = repo.get_canonical_run_matrix(plan_key=test_key)
    assert len(matrix) == 6
    assert all(row["executed"] is False for row in matrix)
    assert all(row["target_ids"] == [target_id] for row in matrix)

    # Simulate executing 4 slots by attaching fake reference_mlflow_run_id
    # We execute all 3 CBOW seeds (42, 43, 44) and 1 SkipGram seed (42).
    # Missing slots will be SkipGram seeds 43 and 44.
    plan_exp_id = f"{exec_plan_id}_e01"
    repo.link_mlflow_run(f"{plan_exp_id}__cbow__s42", "mlflow_run_cbow_42")
    repo.link_mlflow_run(f"{plan_exp_id}__cbow__s43", "mlflow_run_cbow_43")
    repo.link_mlflow_run(f"{plan_exp_id}__cbow__s44", "mlflow_run_cbow_44")
    repo.link_mlflow_run(f"{plan_exp_id}__skipgram__s42", "mlflow_run_sg_42")

    # Verify v_canonical_plan_progress: expected=6, executed=4, missing=2
    progress_after = repo.get_canonical_plan_progress(plan_key=test_key)
    assert len(progress_after) == 1
    p2 = progress_after[0]
    assert p2["expected_slots"] == 6
    assert p2["executed_slots"] == 4
    assert p2["missing_slots"] == 2
    assert pytest.approx(p2["completion_rate"], 0.01) == 4 / 6

    # Verify missing seeds via v_canonical_run_matrix
    matrix_after = repo.get_canonical_run_matrix(plan_key=test_key)
    missing_slots = [r for r in matrix_after if not r["executed"]]
    assert len(missing_slots) == 2
    missing_seeds = {(r["atomic_run_id"], r["seed"]) for r in missing_slots}
    assert missing_seeds == {("skipgram", 43), ("skipgram", 44)}


def test_historical_plan_isolation(repo: CatalogRepository):
    """Ensure past revisions retain their slot history while canonical view reflects only the active revision."""
    test_key = f"hist_test_{uuid.uuid4().hex[:6]}"
    paper_id = f"paper_{uuid.uuid4().hex[:6]}"
    spec_id = f"{test_key}_e01"

    repo.upsert_paper(paper_id=paper_id, title="Historical Test Paper")
    repo.upsert_experiment_spec(
        experiment_spec_id=spec_id,
        paper_id=paper_id,
        name="Scaling Study",
        run_type="trained",
        provenance_status="explicit",
    )

    materializer = CatalogPlanMaterializer(repo)

    # Revision 1: 2 runs (cbow s42, cbow s43)
    plans_r1 = [
        RunPlan(test_key, "e01", os.path.abspath(__file__), "cbow", 42, "cpu"),
        RunPlan(test_key, "e01", os.path.abspath(__file__), "cbow", 43, "cpu"),
    ]
    materializer.materialize_from_plans(
        plan_key=test_key,
        revision=1,
        plans=plans_r1,
        is_canonical=True,
    )

    # Canonical view sees 2 slots for rev 1
    matrix_r1 = repo.get_canonical_run_matrix(plan_key=test_key)
    assert len(matrix_r1) == 2
    assert matrix_r1[0]["plan_revision"] == 1

    # Revision 2: 4 runs (cbow s42, cbow s43, sg s42, sg s43)
    plans_r2 = [
        RunPlan(test_key, "e01", os.path.abspath(__file__), "cbow", 42, "cpu"),
        RunPlan(test_key, "e01", os.path.abspath(__file__), "cbow", 43, "cpu"),
        RunPlan(test_key, "e01", os.path.abspath(__file__), "skipgram", 42, "cpu"),
        RunPlan(test_key, "e01", os.path.abspath(__file__), "skipgram", 43, "cpu"),
    ]
    materializer.materialize_from_plans(
        plan_key=test_key,
        revision=2,
        plans=plans_r2,
        is_canonical=True,  # Automatically demotes rev 1
    )

    # Canonical view sees ONLY 4 slots from rev 2
    matrix_r2 = repo.get_canonical_run_matrix(plan_key=test_key)
    assert len(matrix_r2) == 4
    assert all(r["plan_revision"] == 2 for r in matrix_r2)

    # But rev 1 slots are still physically preserved in planned_run_slots table
    with repo.conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FROM planned_run_slots prs
            JOIN execution_plan_experiments epe ON epe.plan_experiment_id = prs.plan_experiment_id
            WHERE epe.execution_plan_id = %s;
            """,
            (f"{test_key}_r1",),
        )
        count_r1 = cur.fetchone()[0]
        assert count_r1 == 2


def test_resource_tracking_and_substitute_binding(repo: CatalogRepository):
    """Test original required resource preservation vs candidate substitution and canonical binding."""
    paper_id = f"paper_{uuid.uuid4().hex[:6]}"
    spec_id = f"spec_{uuid.uuid4().hex[:6]}"
    plan_key = f"sub_test_{uuid.uuid4().hex[:6]}"

    repo.upsert_paper(paper_id=paper_id, title="Resource Tracking Paper")
    repo.upsert_experiment_spec(
        experiment_spec_id=spec_id,
        paper_id=paper_id,
        name="Scaling on 1B Dataset",
        run_type="trained",
        provenance_status="explicit",
    )

    # 1. Register required resource: Google News 1B (private / unavailable / blocked)
    orig_res_id = "google_news_1b"
    repo.upsert_resource(
        resource_id=orig_res_id,
        kind="dataset",
        name="Google News 1B Tokens",
        access_status="private",
        acquisition_status="unavailable",
        readiness_status="blocked",
        description="Original proprietary Google News corpus mentioned in Mikolov et al. (2013)",
    )
    repo.upsert_resource_source(
        resource_id=orig_res_id,
        source_type="third_party",
        citation="Mikolov et al. 2013",
        is_preferred=True,
    )

    # 2. Register requirement pointing to Google News 1B
    req_id = f"req_data_{uuid.uuid4().hex[:6]}"
    repo.upsert_requirement(
        requirement_id=req_id,
        experiment_spec_id=spec_id,
        role="train_data",
        required_resource_id=orig_res_id,
        required=True,
    )

    # 3. Register candidate substitute: reconstructed CC News 1B
    sub_res_id = "cc_news_1b_reconstructed"
    repo.upsert_resource(
        resource_id=sub_res_id,
        kind="dataset",
        name="Common Crawl 2012 Reconstructed News 1B",
        access_status="public",
        acquisition_status="acquired",
        readiness_status="ready",
        description="Reconstructed 1B news corpus sampled from CC-MAIN-2012",
    )
    repo.upsert_resource_source(
        resource_id=sub_res_id,
        source_type="official_repo",
        url="https://commoncrawl.org",
        is_preferred=True,
    )
    repo.upsert_resource_version(
        resource_version_id="cc_news_1b_v1",
        resource_id=sub_res_id,
        version_label="v1.0",
        size_bytes=4_500_000_000,
        is_verified=True,
    )
    repo.set_resource_canonical_version(sub_res_id, "cc_news_1b_v1")

    # Link candidate substitute to requirement
    repo.upsert_requirement_candidate(
        requirement_id=req_id,
        resource_id=sub_res_id,
        candidate_type="substitute",
        status="selected",
        justification="Google News 1B is private; CC News 2012 provides 1B news tokens with matched vocabulary filter.",
    )

    # 4. Create Execution Plan and bind requirement to substitute version
    plan_id = f"{plan_key}_r1"
    plan_exp_id = f"{plan_id}_exp"
    repo.upsert_execution_plan(
        plan_id, plan_key, revision=1, status="runnable", is_canonical=True
    )
    repo.upsert_plan_experiment(plan_exp_id, plan_id, spec_id)
    repo.bind_plan_requirement(
        plan_experiment_id=plan_exp_id,
        requirement_id=req_id,
        resource_version_id="cc_news_1b_v1",
        binding_type="substitute",
        justification="Binding canonical CC News 1B v1 as verified substitute.",
    )

    # 5. Assert invariants:
    # Original required resource is still Google News 1B, untouched!
    with repo.conn.cursor() as cur:
        cur.execute(
            "SELECT required_resource_id, role FROM experiment_requirements WHERE requirement_id = %s;",
            (req_id,),
        )
        req_row = cur.fetchone()
        assert req_row[0] == "google_news_1b"
        assert req_row[1] == "train_data"

        cur.execute(
            "SELECT access_status, acquisition_status, readiness_status FROM resources WHERE resource_id = 'google_news_1b';"
        )
        orig_row = cur.fetchone()
        assert orig_row == ("private", "unavailable", "blocked")

        # Plan binding points to substitute version with binding_type='substitute'
        cur.execute(
            "SELECT resource_version_id, binding_type, justification FROM execution_plan_bindings WHERE plan_experiment_id = %s AND requirement_id = %s;",
            (plan_exp_id, req_id),
        )
        bind_row = cur.fetchone()
        assert bind_row[0] == "cc_news_1b_v1"
        assert bind_row[1] == "substitute"

    # 6. Verify v_resource_inventory
    inventory = repo.get_resource_inventory()
    inv_map = {r["resource_id"]: r for r in inventory}
    assert "google_news_1b" in inv_map
    assert inv_map["google_news_1b"]["dependent_requirement_count"] == 1
    assert "cc_news_1b_reconstructed" in inv_map
    assert (
        inv_map["cc_news_1b_reconstructed"]["canonical_version_id"] == "cc_news_1b_v1"
    )
    assert (
        inv_map["cc_news_1b_reconstructed"]["preferred_source_url"]
        == "https://commoncrawl.org"
    )


def test_binding_integrity_across_experiments(repo: CatalogRepository):
    """Enforce that execution_plan_bindings rejects binding requirements from a different experiment."""
    paper_id = f"paper_{uuid.uuid4().hex[:6]}"
    spec_a = f"spec_a_{uuid.uuid4().hex[:6]}"
    spec_b = f"spec_b_{uuid.uuid4().hex[:6]}"
    plan_key = f"bind_int_{uuid.uuid4().hex[:6]}"

    repo.upsert_paper(paper_id=paper_id, title="Binding Integrity Paper")
    repo.upsert_experiment_spec(spec_a, paper_id, "Experiment A", "trained", "explicit")
    repo.upsert_experiment_spec(spec_b, paper_id, "Experiment B", "trained", "explicit")

    # Resource and version
    res_id = f"res_{uuid.uuid4().hex[:6]}"
    ver_id = f"ver_{uuid.uuid4().hex[:6]}"
    repo.upsert_resource(
        res_id, "dataset", "Common Dataset", "public", "acquired", "ready"
    )
    repo.upsert_resource_version(ver_id, res_id, "v1")

    # Requirement belonging exclusively to Experiment B
    req_b = f"req_b_{uuid.uuid4().hex[:6]}"
    repo.upsert_requirement(req_b, experiment_spec_id=spec_b, role="evaluation_data")

    # Plan Experiment for Experiment A
    plan_id = f"{plan_key}_r1"
    plan_exp_a = f"{plan_id}_exp_a"
    repo.upsert_execution_plan(
        plan_id, plan_key, revision=1, status="runnable", is_canonical=True
    )
    repo.upsert_plan_experiment(plan_exp_a, plan_id, experiment_spec_id=spec_a)

    # Attempting to bind Experiment B's requirement to Experiment A's plan must fail!
    with pytest.raises(psycopg.Error) as exc_info:
        repo.bind_plan_requirement(
            plan_experiment_id=plan_exp_a,
            requirement_id=req_b,
            resource_version_id=ver_id,
            binding_type="exact",
        )
    assert "Plan binding integrity violation" in str(exc_info.value)


def test_resource_canonical_version_integrity(repo: CatalogRepository):
    """Enforce that resources.canonical_version_id cannot point to a version of a different resource."""
    res_a = f"res_a_{uuid.uuid4().hex[:6]}"
    res_b = f"res_b_{uuid.uuid4().hex[:6]}"
    ver_b = f"ver_b_{uuid.uuid4().hex[:6]}"

    repo.upsert_resource(res_a, "dataset", "Resource A", "public", "acquired", "ready")
    repo.upsert_resource(res_b, "dataset", "Resource B", "public", "acquired", "ready")
    repo.upsert_resource_version(ver_b, resource_id=res_b, version_label="v1")

    # Attempting to set Resource A's canonical version to a version belonging to Resource B must fail!
    with pytest.raises(psycopg.Error):
        repo.set_resource_canonical_version(
            resource_id=res_a, resource_version_id=ver_b
        )
