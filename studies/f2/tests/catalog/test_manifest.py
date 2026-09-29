from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from f2.catalog.manifest import digest_manifest, load_manifest, read_manifest

F2_ROOT = Path(__file__).resolve().parents[2]


def test_manifest_requires_current_schema_version(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be 1"):
        read_manifest(path)


def test_w2v_catalog_manifest_is_self_consistent():
    payload = read_manifest(F2_ROOT / "catalog" / "w2v.json")

    assert {paper["paper_id"] for paper in payload["papers"]} == {"w2v1", "w2v2"}
    assert len(payload["targets"]) == 19
    assert len(payload["experiment_specs"]) == 13
    assert len(payload["execution_plans"]) == 9
    assert len(payload["plan_experiments"]) == 18
    assert len(payload["requirement_candidates"]) == 27
    assert len(payload["resource_bindings"]) == 36
    assert len(payload["planned_run_slots"]) == 306
    assert {slot["seed"] for slot in payload["planned_run_slots"]} == {1, 7, 19}
    assert all(
        slot["parameters"]["requires_approval"] for slot in payload["planned_run_slots"]
    )
    lm1b_reduced = [
        slot
        for slot in payload["planned_run_slots"]
        if slot["planned_run_slot_id"].startswith("w2v2-lm1b-reduced-r1-")
    ]
    assert len(lm1b_reduced) == 18
    assert {slot["parameters"]["training_tokens"] for slot in lm1b_reduced} == {
        791_844_834
    }
    assert digest_manifest(payload) == digest_manifest(
        read_manifest(F2_ROOT / "catalog" / "w2v.json")
    )

    corpus_candidates = {
        row["resource_id"] for row in payload["requirement_candidates"]
    }
    assert corpus_candidates == {
        "f2-wmt-normalized",
        "f2-lm1b-normalized",
        "f2-umbc-normalized",
    }
    corpus_versions = [
        version
        for version in payload["resource_versions"]
        if version["resource_id"]
        in {"f2-wmt-normalized", "f2-lm1b-normalized", "f2-umbc-normalized"}
    ]
    assert len(corpus_versions) == 3
    assert all("checksum" not in version for version in corpus_versions)


def test_manifest_rejects_unknown_requirement_candidate_resource(tmp_path):
    payload = read_manifest(F2_ROOT / "catalog" / "w2v.json")
    payload["requirement_candidates"][0]["resource_id"] = "missing"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown resource_id 'missing'"):
        read_manifest(path)


def test_planned_slots_require_all_verified_resource_bindings(tmp_path):
    payload = read_manifest(F2_ROOT / "catalog" / "w2v.json")
    for version in payload["resource_versions"]:
        if version["resource_version_id"] == "w2v-questions-words-google-code-export":
            version["is_verified"] = False
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="verified immutable resource"):
        read_manifest(path)


def test_planned_slots_reject_missing_required_binding(tmp_path):
    payload = read_manifest(F2_ROOT / "catalog" / "w2v.json")
    payload["resource_bindings"] = payload["resource_bindings"][:1]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="require binding"):
        read_manifest(path)


def test_manifest_rejects_unknown_references(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "targets": [
                    {
                        "target_id": "target",
                        "paper_id": "missing",
                        "location_type": "table",
                        "location_label": "Table 1",
                        "target_type": "metric",
                        "description": "value",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown paper_id 'missing'"):
        read_manifest(path)


def test_resource_version_is_loaded_before_canonical_pointer_is_set():
    repo = MagicMock()
    payload = {
        "schema_version": 1,
        "resources": [
            {
                "resource_id": "corpus",
                "kind": "dataset",
                "name": "Corpus",
                "access_status": "private",
                "acquisition_status": "verified",
                "readiness_status": "ready",
                "canonical_version_id": "corpus-v1",
            }
        ],
        "resource_versions": [
            {
                "resource_version_id": "corpus-v1",
                "resource_id": "corpus",
                "uri": "s3://bucket/release.json",
                "checksum_algo": "sha256",
                "checksum": "a" * 64,
                "is_verified": True,
            }
        ],
    }

    counts = load_manifest(repo, payload)

    assert repo.method_calls[:3] == [
        call.upsert_resource(
            resource_id="corpus",
            kind="dataset",
            name="Corpus",
            access_status="private",
            acquisition_status="verified",
            readiness_status="ready",
            canonical_version_id=None,
        ),
        call.upsert_resource_version(
            resource_version_id="corpus-v1",
            resource_id="corpus",
            uri="s3://bucket/release.json",
            checksum_algo="sha256",
            checksum="a" * 64,
            is_verified=True,
        ),
        call.set_resource_canonical_version("corpus", "corpus-v1"),
    ]
    assert counts["resources"] == counts["resource_versions"] == 1
