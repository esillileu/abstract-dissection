from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path
from typing import Annotated

import typer

from repro_core.context.paths import RuntimePaths

from ...db.repository import CorpusStateRepository
from ...db.session import get_connection


def review_audit(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="Run ID to export audit records for")
    ],
    output_file: Annotated[
        Path | None,
        typer.Option("--output-file", "-o", help="Audit review JSONL output path"),
    ] = None,
    blind: Annotated[
        bool,
        typer.Option(
            "--blind/--no-blind",
            help="Mask model predictions & scores for double-blind auditing",
        ),
    ] = True,
) -> None:
    """Export the audit assignments with text for manual labeling (supports double-blind mode)."""
    paths = RuntimePaths.from_environment()
    target_output_file = output_file or (
        paths.staging_root / "exp" / "f2" / "00_corpus_audit_set_50k_400_blind.jsonl"
    )
    target_output_file.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        audit_items = repo.get_audit_assignments(run_id)
        if not audit_items:
            typer.echo(
                f"No audit assignments found for run '{run_id}'. Run 'repro f2 corpus audit' first."
            )
            return

        # Map URLs to clean text snippet if shards exist
        shard_candidates = [
            paths.staging_root / "exp" / "f2" / "confirmatory_50k" / "clean_shards",
            paths.staging_root / "exp" / "f2" / "sample_10k" / "clean_shards",
            paths.staging_root / "exp" / "f2" / "sample" / "clean_shards",
        ]
        shard_dir = next(
            (p for p in shard_candidates if p.exists()), shard_candidates[0]
        )

        url_to_text: dict[str, str] = {}
        if shard_dir.exists():
            for sf in shard_dir.glob("*.txt"):
                docs = re.findall(
                    r'<DOC url="(.*?)" words="(\d+)">\n(.*?)\n</DOC>',
                    sf.read_text(encoding="utf-8"),
                    re.DOTALL,
                )
                for u, _, txt in docs:
                    url_to_text[u] = txt.strip()

        exported_items = []
        with target_output_file.open("w", encoding="utf-8") as f:
            for item in audit_items:
                u = item["url"]
                domain = urllib.parse.urlparse(u).netloc.lower()
                snippet = url_to_text.get(u, "")
                if snippet and len(snippet) > 800:
                    snippet = snippet[:800] + "..."

                if blind:
                    review_entry = {
                        "audit_id": item["audit_id"],
                        "candidate_id": item["candidate_id"],
                        "priority_order": item["priority_order"],
                        "wave": item.get("wave", 1),
                        "crawl_id": item["crawl_id"],
                        "url": u,
                        "domain": domain,
                        "word_count": item["word_count"],
                        "text_snippet": snippet,
                        "gold_class": item["gold_class"]
                        if item["is_audited"]
                        else None,
                        "word_count_gold": item["word_count_gold"]
                        if item["is_audited"]
                        else None,
                        "auditor_id": item.get("auditor_id"),
                        "notes": item.get("notes"),
                    }
                else:
                    review_entry = {
                        "audit_id": item["audit_id"],
                        "candidate_id": item["candidate_id"],
                        "priority_order": item["priority_order"],
                        "design_stratum": item.get("design_stratum"),
                        "audit_stratum": item["audit_stratum"],
                        "wave": item.get("wave", 1),
                        "crawl_id": item["crawl_id"],
                        "url": u,
                        "domain": domain,
                        "first_stage_inclusion_probability": item["first_stage_pi"],
                        "first_stage_design_weight": item["first_stage_weight"],
                        "audit_inclusion_probability": item[
                            "audit_inclusion_probability"
                        ],
                        "audit_design_weight": item["audit_design_weight"],
                        "news_score": item["news_score"],
                        "is_news_predicted": item["is_news_predicted"],
                        "is_english": item["is_english"],
                        "is_valid": item["is_valid"],
                        "word_count": item["word_count"],
                        "word_count_proxy": item["word_count_proxy"],
                        "diagnostics": item["diagnostics"],
                        "text_snippet": snippet,
                        "gold_class": item["gold_class"]
                        if item["is_audited"]
                        else None,
                        "word_count_gold": item["word_count_gold"]
                        if item["is_audited"]
                        else None,
                        "auditor_id": item.get("auditor_id"),
                        "notes": item.get("notes"),
                    }
                exported_items.append(review_entry)
                f.write(json.dumps(review_entry, default=str) + "\n")

        # Also write Markdown Review Dossier
        md_file = target_output_file.parent / "00_corpus_audit_set_50k_400_review.md"
        mode_tag = "Double-Blind Mode" if blind else "Unblinded Control Mode"
        md_lines = [
            f"# Phase-2 8-Stratum Gold Audit Set ({len(audit_items)} Documents) — Run `{run_id}` ({mode_tag})",
            "",
            "> **Pre-specified 8-Stratum Factorial Audit Design (Crawl x Prefilter x Postfilter)**",
            "",
            "---",
            "",
        ]
        for entry in exported_items:
            strat_display = (
                f"Stratum `{entry.get('design_stratum', 'N/A')}`"
                if not blind
                else "Blinded Unit"
            )
            md_lines.extend(
                [
                    f"### #{entry['priority_order']:03d} [{strat_display}] `{entry['audit_id']}`",
                    f"- **URL:** [{entry['url']}]({entry['url']})",
                    f"- **Crawl:** `{entry['crawl_id']}` | **Domain:** `{entry['domain']}`",
                    f"- **Document Length:** `{entry.get('word_count', 0):,} words`",
                ]
            )
            if not blind:
                md_lines.extend(
                    [
                        f"- **Classifier Score:** `{entry.get('news_score', 0.0):.1f}` | **Proxy Words:** `{entry.get('word_count_proxy', 0):,}`",
                        f"- **Weights:** $\\pi_1 = {entry.get('first_stage_inclusion_probability', 0.0):.2e}$ | $\\pi_2 = {entry.get('audit_inclusion_probability', 0.0):.4f}$",
                    ]
                )
            if entry["text_snippet"]:
                md_lines.extend(["", "```text", entry["text_snippet"], "```", ""])
            else:
                md_lines.extend(["", "*(No clean text snippet retained)*", ""])
            md_lines.extend(["---", ""])

        md_file.write_text("\n".join(md_lines), encoding="utf-8")
        typer.echo(f"Exported {len(audit_items)} audit review documents to:")
        typer.echo(f"  - JSONL: {target_output_file}")
        typer.echo(f"  - Markdown Dossier: {md_file}")
