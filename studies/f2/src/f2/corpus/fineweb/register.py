"""Registration of materialized FineWeb shards into SeaweedFS S3 and PostgreSQL catalog."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

import psycopg
import requests
from repro_io.s3 import S3ObjectStore
from tqdm import tqdm

from f2.catalog.db.repository import CatalogRepository
from f2.corpus.db.repository import CorpusStateRepository
from f2.corpus.sources import stable_id


def load_and_validate_checkpoint(
    checkpoint_path: Path,
    shards_dir: Path,
    max_shards: int | None = None,
) -> list[dict[str, Any]]:
    """Load checkpoint.json and validate shards against local disk."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"FineWeb checkpoint not found: {checkpoint_path}")

    with checkpoint_path.open("r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    completed = checkpoint.get("completed_shards", [])
    if max_shards is not None and max_shards > 0:
        if len(completed) < max_shards:
            raise ValueError(
                f"Checkpoint contains only {len(completed)} shards; requested {max_shards}"
            )
        shards: list[dict[str, Any]] = completed[:max_shards]
    else:
        shards = completed

    # Validate indices and files
    for idx, shard in enumerate(shards):
        if shard["index"] != idx:
            raise ValueError(
                f"Shard index mismatch: expected {idx}, found {shard['index']}"
            )

        shard_filename = f"shard-{idx:05d}.txt.zst"
        local_path = shards_dir / shard_filename
        if not local_path.is_file():
            raise FileNotFoundError(f"Shard file not found on disk: {local_path}")

        file_size = local_path.stat().st_size
        if file_size != shard["compressed_bytes"]:
            raise ValueError(
                f"Shard {idx} size mismatch: disk={file_size}, checkpoint={shard['compressed_bytes']}"
            )

    return shards


def _object_exists(
    store: S3ObjectStore, uri: str, expected_size: int | None = None
) -> bool:
    try:
        meta = store.head(uri)
        if expected_size is not None:
            return meta.byte_size == expected_size
        return True
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return False
        raise


def upload_shards_to_s3(
    shards: list[dict[str, Any]],
    shards_dir: Path,
    store: S3ObjectStore,
    *,
    prefix: str = "processed/fineweb/normalized",
    manifest_prefix: str = "manifests/fineweb/normalized",
    max_workers: int = 8,
) -> int:
    """Upload verified shards and manifest to SeaweedFS S3 concurrently."""
    store.probe()

    try:
        existing_sizes = {
            m.uri: m.byte_size for m in store.list(store.uri(f"{prefix}/"))
        }
    except Exception:
        existing_sizes = {}

    tasks: list[tuple[Path, str, int]] = []
    for shard in shards:
        idx = shard["index"]
        filename = f"shard-{idx:05d}.txt.zst"
        local_path = shards_dir / filename
        s3_uri = store.uri(f"{prefix}/{filename}")
        tasks.append((local_path, s3_uri, shard["compressed_bytes"]))

    def _worker(item: tuple[Path, str, int]) -> bool:
        path, uri, size = item
        if uri in existing_sizes and existing_sizes[uri] == size:
            return False
        if uri not in existing_sizes and _object_exists(store, uri, expected_size=size):
            return False
        store.put_file(path, uri)
        return True

    uploaded_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        with tqdm(
            total=len(tasks),
            desc="Uploading FineWeb shards to S3",
            unit="shard",
            ncols=90,
        ) as bar:
            for fut in concurrent.futures.as_completed(futures):
                if fut.result():
                    uploaded_count += 1
                bar.update(1)

    # Write and upload manifest
    manifest_payload = {
        "dataset": "HuggingFaceFW/fineweb",
        "dump": "CC-MAIN-2013-20",
        "stage": "word2vec_normalized",
        "shard_count": len(shards),
        "total_words": sum(s["word_count"] for s in shards),
        "total_tokens": sum(
            s.get("train_words_count", s["word_count"] + s.get("newline_count", 0))
            for s in shards
        ),
        "total_records": sum(s["record_count"] for s in shards),
        "total_compressed_bytes": sum(s["compressed_bytes"] for s in shards),
        "shards": shards,
    }
    temp_manifest = shards_dir.parent / "manifest_registered.json"
    temp_manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    try:
        store.put_file(temp_manifest, store.uri(f"{manifest_prefix}/manifest.json"))
    finally:
        temp_manifest.unlink(missing_ok=True)

    return uploaded_count


def register_corpus_in_db(
    shards: list[dict[str, Any]],
    conn: psycopg.Connection,
    store: S3ObjectStore,
    *,
    resource_id: str = "f2-fineweb-normalized",
    resource_version_id: str = "f2-fineweb-2013-news-normalized-v1",
    version_label: str = "v1",
    prefix: str = "processed/fineweb/normalized",
    dump: str = "CC-MAIN-2013-20",
) -> dict[str, int]:
    """Register corpus resources, resource versions, artifacts, shards, and statistics in DB."""
    total_words = sum(s["word_count"] for s in shards)
    total_newlines = sum(s.get("newline_count", 0) for s in shards)
    total_tokens = sum(
        s.get("train_words_count", s["word_count"] + s.get("newline_count", 0))
        for s in shards
    )
    total_records = sum(s["record_count"] for s in shards)
    total_bytes = sum(s["compressed_bytes"] for s in shards)

    repo = CorpusStateRepository(conn)

    with conn.transaction():
        with conn.cursor() as cur:
            # 1. catalog.resources (initial insert without canonical_version_id)
            cur.execute(
                """
                INSERT INTO catalog.resources (
                    resource_id, kind, name, description, access_status,
                    acquisition_status, readiness_status, notes
                ) VALUES (%s, 'dataset', %s, %s, 'public', 'verified', 'ready', %s)
                ON CONFLICT (resource_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    access_status = EXCLUDED.access_status,
                    acquisition_status = EXCLUDED.acquisition_status,
                    readiness_status = EXCLUDED.readiness_status,
                    notes = EXCLUDED.notes;
                """,
                (
                    resource_id,
                    "FineWeb 2013 News Crawl, Word2Vec-normalized",
                    "Verified reconstructed news training corpus from FineWeb CC-MAIN-2013-20 with deterministic record-preserving shard order.",
                    f"FineWeb 2013 news-filtered domain substitute for Google News; {len(shards)} shards registered.",
                ),
            )

            # 2. catalog.resource_versions
            metadata = {
                "representation": "normalized",
                "shard_count": len(shards),
                "shard_order": f"ascending shard_index 0..{len(shards) - 1}",
                "lexical_words": total_words,
                "training_token_budget": total_words,
                "boundary_policy": "document_paragraph_lines",
                "validation_profile": "word2vec-2013-compatibility-v1",
                "validation_verdict": "PASS",
                "paper_compatibility": "compatible_reconstruction",
                "dump": dump,
            }
            cur.execute(
                """
                INSERT INTO catalog.resource_versions (
                    resource_version_id, resource_id, version_label, uri, size_bytes, metadata, is_verified, notes
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, TRUE, %s)
                ON CONFLICT (resource_version_id) DO UPDATE SET
                    resource_id = EXCLUDED.resource_id,
                    version_label = EXCLUDED.version_label,
                    uri = EXCLUDED.uri,
                    size_bytes = EXCLUDED.size_bytes,
                    metadata = EXCLUDED.metadata,
                    is_verified = EXCLUDED.is_verified,
                    notes = EXCLUDED.notes;
                """,
                (
                    resource_version_id,
                    resource_id,
                    version_label,
                    store.uri(f"{prefix}/"),
                    total_bytes,
                    json.dumps(metadata),
                    f"Verified {len(shards)}-shard prefix of FineWeb 2013 surrogate corpus.",
                ),
            )

            # 3. catalog.resources update canonical_version_id
            cur.execute(
                """
                UPDATE catalog.resources
                SET canonical_version_id = %s
                WHERE resource_id = %s;
                """,
                (resource_version_id, resource_id),
            )

        # 3. artifacts and corpus_shards
        for s in tqdm(
            shards,
            desc="Registering shards in DB",
            unit="shard",
            ncols=90,
        ):
            artifact_id = "normalized-" + stable_id(
                resource_version_id, s["index"], s["physical_sha256"]
            )
            s3_uri = store.uri(f"{prefix}/shard-{s['index']:05d}.txt.zst")
            repo.register_artifact(
                artifact_id=artifact_id,
                stage="normalized",
                s3_uri=s3_uri,
                sha256=s["physical_sha256"],
                byte_size=s["compressed_bytes"],
                format="txt.zst",
                record_count=s["record_count"],
                resource_version_id=resource_version_id,
                integrity_status="verified",
                verification_report=s,
            )
            repo.register_corpus_shard(
                resource_version_id=resource_version_id,
                shard_index=s["index"],
                artifact_id=artifact_id,
                word_count=s["word_count"],
                doc_count=s["record_count"],
                byte_size=s["compressed_bytes"],
            )

        # 4. corpus_version_stats
        repo.upsert_corpus_version_stats(
            resource_version_id=resource_version_id,
            total_words=total_words,
            total_tokens=total_tokens,
            total_documents=total_records,
            total_sentences=total_newlines,
            total_bytes=total_bytes,
            total_shards=len(shards),
            source_composition={
                "fineweb": total_words,
                "metrics": {
                    "normalized_lexical_words": total_words,
                    "newlines": total_newlines,
                    "word2vec_train_words": total_tokens,
                },
            },
            year_distribution={"2013": total_words},
        )

    return {
        "shards": len(shards),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_records": total_records,
        "total_bytes": total_bytes,
    }


def register_execution_plan(
    conn: psycopg.Connection,
    resource_version_id: str = "f2-fineweb-2013-news-normalized-v1",
    plan_id: str = "w2v1-fineweb-reconstruction-r1",
) -> dict[str, int]:
    """Register FineWeb execution plan, experiments, bindings, and run slots."""
    catalog = CatalogRepository(conn)
    seeds = [1, 7, 19]

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SET search_path = catalog, corpus, public")

        # 1. Execution plan
        catalog.upsert_execution_plan(
            execution_plan_id=plan_id,
            plan_key="w2v1-fineweb-reconstruction",
            revision=1,
            status="runnable",
            source_ref="studies/f2/src/f2/corpus/fineweb/register.py",
            is_canonical=False,
            notes="FineWeb 2013 surrogate reconstruction plan for W2V1 Table 2, Table 3, and Table 6.",
        )

        # 2. Plan experiments & bindings
        # Table 2 CBOW
        exp_table2 = f"{plan_id}-w2v1-table2-cbow"
        catalog.upsert_plan_experiment(
            plan_experiment_id=exp_table2,
            execution_plan_id=plan_id,
            experiment_spec_id="w2v1-table2-cbow",
            enabled=True,
            parameters={"classification": "reconstruction", "corpus": "fineweb"},
            notes="W2V1 Table 2 architecture scaling with FineWeb 2013 news surrogate.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table2,
            requirement_id="w2v1-table2-train-data",
            resource_version_id=resource_version_id,
            binding_type="substitute",
            justification="FineWeb 2013 news surrogate corpus satisfies all Table 2 budgets.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table2,
            requirement_id="w2v1-table2-eval",
            resource_version_id="w2v-questions-words-google-code-export",
            binding_type="exact",
            justification="Pinned verified evaluation resource.",
        )

        # Table 3 CBOW
        exp_table3_cbow = f"{plan_id}-w2v1-table3-cbow"
        catalog.upsert_plan_experiment(
            plan_experiment_id=exp_table3_cbow,
            execution_plan_id=plan_id,
            experiment_spec_id="w2v1-table3-cbow",
            enabled=True,
            parameters={"classification": "reconstruction", "corpus": "fineweb"},
            notes="W2V1 Table 3 CBOW comparison with FineWeb 2013 news surrogate.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table3_cbow,
            requirement_id="w2v1-table3-cbow-train",
            resource_version_id=resource_version_id,
            binding_type="substitute",
            justification="FineWeb 2013 news surrogate corpus satisfies Table 3 320M budget.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table3_cbow,
            requirement_id="w2v1-table3-cbow-word-eval",
            resource_version_id="w2v-questions-words-google-code-export",
            binding_type="exact",
            justification="Pinned verified evaluation resource.",
        )

        # Table 3 Skipgram
        exp_table3_sg = f"{plan_id}-w2v1-table3-skipgram"
        catalog.upsert_plan_experiment(
            plan_experiment_id=exp_table3_sg,
            execution_plan_id=plan_id,
            experiment_spec_id="w2v1-table3-skipgram",
            enabled=True,
            parameters={"classification": "reconstruction", "corpus": "fineweb"},
            notes="W2V1 Table 3 Skip-gram comparison with FineWeb 2013 news surrogate.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table3_sg,
            requirement_id="w2v1-table3-sg-train",
            resource_version_id=resource_version_id,
            binding_type="substitute",
            justification="FineWeb 2013 news surrogate corpus satisfies Table 3 320M budget.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table3_sg,
            requirement_id="w2v1-table3-sg-word-eval",
            resource_version_id="w2v-questions-words-google-code-export",
            binding_type="exact",
            justification="Pinned verified evaluation resource.",
        )

        # Table 6 CBOW (6B words)
        exp_table6_cbow = f"{plan_id}-w2v1-table6-cbow-6b"
        catalog.upsert_plan_experiment(
            plan_experiment_id=exp_table6_cbow,
            execution_plan_id=plan_id,
            experiment_spec_id="w2v1-table6-cbow-6b",
            enabled=True,
            parameters={"classification": "reconstruction", "corpus": "fineweb"},
            notes="W2V1 Table 6 CBOW scale (6B words) with FineWeb 2013 news surrogate.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table6_cbow,
            requirement_id="w2v1-table6-cbow-train",
            resource_version_id=resource_version_id,
            binding_type="substitute",
            justification="FineWeb 2013 6B prefix satisfies Table 6 6B token budget.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table6_cbow,
            requirement_id="w2v1-table6-cbow-eval",
            resource_version_id="w2v-questions-words-google-code-export",
            binding_type="exact",
            justification="Pinned verified evaluation resource.",
        )

        # Table 6 Skipgram (6B words)
        exp_table6_sg = f"{plan_id}-w2v1-table6-skipgram-6b"
        catalog.upsert_plan_experiment(
            plan_experiment_id=exp_table6_sg,
            execution_plan_id=plan_id,
            experiment_spec_id="w2v1-table6-skipgram-6b",
            enabled=True,
            parameters={"classification": "reconstruction", "corpus": "fineweb"},
            notes="W2V1 Table 6 Skip-gram scale (6B words) with FineWeb 2013 news surrogate.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table6_sg,
            requirement_id="w2v1-table6-sg-train",
            resource_version_id=resource_version_id,
            binding_type="substitute",
            justification="FineWeb 2013 6B prefix satisfies Table 6 6B token budget.",
        )
        catalog.bind_plan_requirement(
            plan_experiment_id=exp_table6_sg,
            requirement_id="w2v1-table6-sg-eval",
            resource_version_id="w2v-questions-words-google-code-export",
            binding_type="exact",
            justification="Pinned verified evaluation resource.",
        )

        # Our Table 4 NNLM uses the same 6B surrogate; its 100d run also supplies Table 6.
        from f2.definition import DEFINITION
        from f2.suites.w2v1.spec import parse_run_spec

        exp_nnlm = f"{plan_id}-w2v1-google-news-nnlm-6b"
        config_path = DEFINITION.get_suite("w2v1").config_root / "e03_table4.yaml"
        nnlm_configs = [
            parse_run_spec(
                config_path, atomic_run_id=f"fineweb--nnlm-d{dim}-w6000m"
            ).to_executor_config()
            for dim in (20, 50, 100)
        ]
        catalog.upsert_plan_experiment(
            plan_experiment_id=exp_nnlm,
            execution_plan_id=plan_id,
            experiment_spec_id="w2v1-google-news-nnlm-6b",
            enabled=True,
            parameters={
                "classification": "reconstruction",
                "corpus": "fineweb",
                "training": nnlm_configs[0]["training"],
                "distribution": nnlm_configs[0]["distribution"],
                "dimensions": [20, 50, 100],
            },
            notes="Our Table 4 NNLM 6B; unspecified history/hidden/optimizer settings are explicit reconstruction decisions in YAML and catalog notes. Table 6 reuses 100d.",
        )
        for requirement, version, binding, justification in (
            (
                "w2v1-nnlm-6b-train",
                resource_version_id,
                "substitute",
                "Same FineWeb 2013 news surrogate and exact prefix policy as existing W2V1 6B runs.",
            ),
            (
                "w2v1-google-news-eval-nnlm",
                "w2v-questions-words-google-code-export",
                "exact",
                "Pinned verified full-vocabulary benchmark.",
            ),
        ):
            catalog.bind_plan_requirement(
                plan_experiment_id=exp_nnlm,
                requirement_id=requirement,
                resource_version_id=version,
                binding_type=binding,
                justification=justification,
            )

        # 3. Generate planned run slots
        slot_count = 0

        # Table 2: 4 dimensions x 6 budgets x 3 seeds = 72 slots
        dimensions = [50, 100, 300, 600]
        budgets = [
            24_000_000,
            49_000_000,
            98_000_000,
            196_000_000,
            391_000_000,
            783_000_000,
        ]
        for dim in dimensions:
            for budget in budgets:
                cond_id = f"d{dim}-w{budget // 1_000_000}m"
                for seed in seeds:
                    slot_key = f"{cond_id}-s{seed}"
                    slot_id = f"{plan_id}-{slot_key}"
                    catalog.upsert_planned_run_slot(
                        planned_run_slot_id=slot_id,
                        plan_experiment_id=exp_table2,
                        slot_key=slot_key,
                        atomic_run_id=cond_id,
                        variant_key=cond_id,
                        seed=seed,
                        parameters={
                            "device": "cpu",
                            "epochs": 3,
                            "threads": 1,
                            "classification": "reconstruction",
                            "training_tokens": budget,
                            "embedding_dimension": dim,
                            "requires_approval": True,
                        },
                        expected=True,
                        notes="FineWeb Table 2 reconstruction slot.",
                    )
                    slot_count += 1

        # Table 3: 2 conditions x 3 seeds = 6 slots
        for model_kind, exp_id in [
            ("cbow", exp_table3_cbow),
            ("skipgram", exp_table3_sg),
        ]:
            cond_id = f"{model_kind}-d640-w320m"
            for seed in seeds:
                slot_key = f"{cond_id}-s{seed}"
                slot_id = f"{plan_id}-{slot_key}"
                catalog.upsert_planned_run_slot(
                    planned_run_slot_id=slot_id,
                    plan_experiment_id=exp_id,
                    slot_key=slot_key,
                    atomic_run_id=cond_id,
                    variant_key=cond_id,
                    seed=seed,
                    parameters={
                        "device": "cpu",
                        "epochs": 3,
                        "threads": 1,
                        "classification": "reconstruction",
                        "training_tokens": 320_000_000,
                        "embedding_dimension": 640,
                        "requires_approval": True,
                    },
                    expected=True,
                    notes=f"FineWeb Table 3 {model_kind} reconstruction slot.",
                )
                slot_count += 1

        # Table 6: 2 conditions x 3 seeds = 6 slots
        for model_kind, exp_id in [
            ("cbow", exp_table6_cbow),
            ("skipgram", exp_table6_sg),
        ]:
            cond_id = f"{model_kind}-d1000-w6000m"
            for seed in seeds:
                slot_key = f"{cond_id}-s{seed}"
                slot_id = f"{plan_id}-{slot_key}"
                catalog.upsert_planned_run_slot(
                    planned_run_slot_id=slot_id,
                    plan_experiment_id=exp_id,
                    slot_key=slot_key,
                    atomic_run_id=cond_id,
                    variant_key=cond_id,
                    seed=seed,
                    parameters={
                        "device": "cpu",
                        "epochs": 1,
                        "threads": 1,
                        "classification": "reconstruction",
                        "training_tokens": 6_000_000_000,
                        "embedding_dimension": 1000,
                        "requires_approval": True,
                    },
                    expected=True,
                    notes=f"FineWeb Table 6 {model_kind} 6B reconstruction slot.",
                )
                slot_count += 1

        for config in nnlm_configs:
            condition = str(config["atomic_run_id"]).split("--", 1)[1]
            for seed in seeds:
                catalog.upsert_planned_run_slot(
                    planned_run_slot_id=f"{plan_id}-{condition}-s{seed}",
                    plan_experiment_id=exp_nnlm,
                    slot_key=f"{condition}-s{seed}",
                    atomic_run_id=condition,
                    variant_key=condition,
                    seed=seed,
                    parameters={
                        "device": "cpu",
                        "epochs": 3,
                        "threads": 14,
                        "classification": "reconstruction",
                        "training_tokens": 6_000_000_000,
                        "embedding_dimension": config["training"][
                            "embedding_dimension"
                        ],
                        "training": config["training"],
                        "distribution": config["distribution"],
                        "requires_approval": True,
                    },
                    expected=True,
                    notes="Table 4 Our NNLM; canonical 100d is reused by Table 6.",
                )
                slot_count += 1

    return {
        "execution_plan_id": plan_id,
        "experiments_count": 6,
        "planned_slots_count": slot_count,
    }


__all__ = [
    "load_and_validate_checkpoint",
    "register_corpus_in_db",
    "register_execution_plan",
    "upload_shards_to_s3",
]
