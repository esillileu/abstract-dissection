from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

from dlfs.analysis.input import AnalysisRun, StudyAnalysisInput
from dlfs.analysis.paths import result_stem
from dlfs.analysis.summary import (
    summary_declarations,
    write_study_summary,
)
from dlfs.identity import Variant, Volume


def _render_studies(
    client,
    output_dir: Path,
    artifact_cache_dir: Path,
    tracking_uri: str,
    studies,
    selected_studies: list[str],
    variants: tuple[Variant, ...],
    render_inputs: dict[tuple[str, Variant], list[AnalysisRun]],
    volume: Volume,
    error_style: str,
    summary_metrics,
    print_summary: bool,
    seed: int | None,
    filename_suffix: str,
    cache_dir: Path,
    refresh_raw: bool = False,
    refresh_analysis: bool = False,
) -> list[Path]:
    renderer = importlib.import_module(f"dlfs.{volume.value}.analysis.render")
    outputs = []
    for study_id in selected_studies:
        if study_id not in renderer.RENDERERS:
            continue
        sources = renderer.STUDY_SOURCES.get(study_id, (study_id,))
        declaration = (
            studies[study_id]
            if sources == (study_id,)
            else type(studies[sources[0]])(
                study_id,
                tuple(
                    condition
                    for source in sources
                    for condition in studies[source].conditions
                ),
            )
        )
        for variant in variants:
            selected_runs = tuple(
                run
                for source in sources
                for run in render_inputs.get((source, variant), ())
            )
            if not selected_runs:
                print(
                    f"skipping {volume.value}/{study_id}: no FINISHED "
                    f"{variant.value} runs",
                    file=sys.stderr,
                )
                continue
            data = StudyAnalysisInput(
                client,
                declaration,
                variant,
                selected_runs,
                cache_dir=artifact_cache_dir,
                tracking_uri=tracking_uri,
                refresh_raw=refresh_raw,
                prepared_cache_dir=(cache_dir / "prepared" / study_id / variant.value),
                refresh_analysis=refresh_analysis,
            )
            output_variants = variants if len(variants) == 1 else (variant,)
            output = output_dir / (
                f"{result_stem(volume, study_id, output_variants)}{filename_suffix}.png"
            )
            rendered = renderer.render_study(
                data,
                study_id,
                output,
                error_style=error_style,
            )
            cached_rendered = []
            for rendered_path in rendered:
                if rendered_path.suffix.lower() in {".png", ".md"}:
                    outputs.append(rendered_path)
                    continue
                cache_output = cache_dir / "render" / rendered_path.name
                cache_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(rendered_path), str(cache_output))
                cached_rendered.append(cache_output)
            summary_path = write_study_summary(
                data,
                volume=volume,
                study_id=study_id,
                metrics=summary_declarations(study_id, summary_metrics),
                output_dir=output_dir,
                output_variants=output_variants,
                print_console=print_summary,
                filename_suffix=filename_suffix,
                cache_dir=cache_dir,
            )
            append_markdown = getattr(renderer, "MARKDOWN_APPENDERS", {}).get(study_id)
            text_reports = [
                path for path in cached_rendered if path.suffix.lower() == ".txt"
            ]
            if append_markdown is not None and text_reports:
                append_markdown(summary_path, text_reports[0], seed=seed)
            data.commit_prepared()
            outputs.append(summary_path)
    return outputs
