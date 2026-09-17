"""Common Crawl acquisition, audit, analysis, and release-building CLI."""

from __future__ import annotations

import typer
from repro_io.commoncrawl.fetcher import RangeFetcher

from .admin import migrate_db
from .analysis import analyze_corpus, calibrate_filters
from .audit import create_audit, record_audit
from .audit_review import review_audit
from .export import build_corpus, export_run
from .index import ensure_cluster_index, validate_connection
from .plan import plan_corpus
from .sample import sample_corpus

app = typer.Typer(
    name="corpus",
    help="Sample, audit, analyze, and publish Common Crawl corpus releases.",
    no_args_is_help=True,
)

app.command("migrate")(migrate_db)
app.command("plan")(plan_corpus)
app.command("sample")(sample_corpus)
app.command("export")(export_run)
app.command("audit")(create_audit)
app.command("audit-review")(review_audit)
app.command("audit-record")(record_audit)
app.command("analyze")(analyze_corpus)
app.command("calibrate")(calibrate_filters)
app.command("build")(build_corpus)

__all__ = [
    "RangeFetcher",
    "analyze_corpus",
    "app",
    "build_corpus",
    "calibrate_filters",
    "create_audit",
    "ensure_cluster_index",
    "export_run",
    "migrate_db",
    "plan_corpus",
    "record_audit",
    "review_audit",
    "sample_corpus",
    "validate_connection",
]
