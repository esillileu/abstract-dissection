from __future__ import annotations

from typing import Annotated

import typer

from ..discovery import SEED_DOMAIN_CATALOG


def plan_corpus(
    crawl: Annotated[
        str, typer.Option("--crawl", "-c", help="Common Crawl ID (e.g. CC-MAIN-2012)")
    ] = "CC-MAIN-2012",
    sample_size: Annotated[
        int, typer.Option("--sample-size", "-n", help="Target sample size")
    ] = 100,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 20260227,
) -> None:
    """Inspect sampling plan, strata parameters, and design inclusion probabilities."""
    typer.echo(f"=== F2 Corpus Sampling Plan ({crawl}) ===")
    typer.echo(f"Random Seed: {seed}")
    typer.echo(f"Target Sample Size: {sample_size}")
    typer.echo("\nCurated Seed Strata:")
    for stratum, domains in SEED_DOMAIN_CATALOG.items():
        typer.echo(
            f"  - {stratum.value}: {len(domains)} domains ({', '.join(domains[:3])}...)"
        )

    typer.echo("\nTwo-Stage Probability Sampling Protocol:")
    typer.echo(
        "  Top-Level Strata: Independent crawl snapshots (CC-MAIN-2009-2010, CC-MAIN-2012)"
    )
    typer.echo("  Stage 1: SRS of m CDX blocks (p_k = m / K)")
    typer.echo(
        "  Stage 2: SRS of n_k records per block (pi_within = n_k / M_k, where M_k = 3,000)"
    )
    typer.echo("  Pre-Fetch Filter: Rule 1 Only (32 binary media extensions)")
    typer.echo("  Reject Exploration Rate: 5% (pi_explore = 0.05)")
    typer.echo("  State Store: PostgreSQL operational control plane")
