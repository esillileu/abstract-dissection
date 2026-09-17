from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from typing import Annotated

import typer

from repro_core.context.paths import RuntimePaths

from .sample_exec import execute_sampling


def sample_corpus(
    crawls: Annotated[
        str, typer.Option("--crawls", "-c", help="Comma-separated crawl IDs")
    ] = "CC-MAIN-2009-2010,CC-MAIN-2012",
    sample_size: Annotated[
        int, typer.Option("--sample-size", "-n", help="Total records to sample")
    ] = 50000,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 20260227,
    prefetch_rule: Annotated[
        str, typer.Option("--prefetch-rule", help="Pre-fetch rule ('rule1' or 'none')")
    ] = "rule1",
    reject_exploration_rate: Annotated[
        float,
        typer.Option(
            "--reject-exploration-rate", help="Sampling rate for rejected records"
        ),
    ] = 0.05,
    bandwidth_limit: Annotated[
        float, typer.Option("--bandwidth-limit", "-b", help="Bandwidth limit in Mbps")
    ] = 20.0,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-j", help="Concurrent fetch workers")
    ] = 4,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Output directory")
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Explicit run identifier for resuming"),
    ] = None,
) -> None:
    """Execute bounded, rate-limited probability sample against Common Crawl backed by PostgreSQL."""
    paths = RuntimePaths.from_environment()
    target_output_dir = output_dir or (paths.staging_root / "exp" / "f2" / "sample")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    active_run_id = run_id or f"run_{seed}_{uuid.uuid4().hex[:8]}"

    crawl_list = [c.strip() for c in crawls.split(",") if c.strip()]
    per_crawl_target = max(1, sample_size // len(crawl_list))

    try:
        exec_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        exec_sha = "unknown"

    config_dict = {
        "seed": seed,
        "sample_size": sample_size,
        "crawls": crawl_list,
        "prefetch_rule": prefetch_rule,
        "reject_exploration_rate": reject_exploration_rate,
        "concurrency": concurrency,
        "bandwidth_limit": bandwidth_limit,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True).encode()
    ).hexdigest()[:16]

    run_meta = {
        "baseline_10k_commit_sha": "f3dee9676517d9a7506b162aff83a111f45209dc",
        "execution_50k_commit_sha": exec_sha,
        "config_hash": config_hash,
        "frozen_parameters": config_dict,
    }

    typer.echo(f"Initializing run '{active_run_id}' in PostgreSQL...")
    typer.echo(f"  - Baseline SHA: {run_meta['baseline_10k_commit_sha']}")
    typer.echo(f"  - Execution SHA: {run_meta['execution_50k_commit_sha']}")
    typer.echo(f"  - Config Hash: {config_hash}")

    execute_sampling(
        active_run_id=active_run_id,
        crawl_list=crawl_list,
        sample_size=sample_size,
        per_crawl_target=per_crawl_target,
        seed=seed,
        bandwidth_limit=bandwidth_limit,
        concurrency=concurrency,
        prefetch_rule=prefetch_rule,
        reject_exploration_rate=reject_exploration_rate,
        target_output_dir=target_output_dir,
        output_dir=output_dir,
        run_meta=run_meta,
    )
