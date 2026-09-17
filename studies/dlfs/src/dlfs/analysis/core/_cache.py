"""Persistent file-keyed analysis cache for run-selection manifests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from repro_core.context.paths import WorkspacePaths

from ._client import AnalysisClient, completed_seed_runs

ANALYSIS_CACHE_SCHEMA_VERSION = 6


def analysis_cache_path(output: Path) -> Path:
    """Keep cache metadata out of the durable artifact tree."""
    paths = WorkspacePaths.from_environment(Path.cwd())
    try:
        relative = output.resolve().relative_to(paths.results_root.resolve())
    except ValueError:
        return output.with_name(f".{output.name}.analysis-cache.json")
    return (paths.cache_root / "exp" / relative).with_name(
        f".{output.name}.analysis-cache.json"
    )


def cached_analysis_outputs(
    client: AnalysisClient,
    output: Path,
) -> list[Path] | None:
    """Return existing outputs when every recorded selection has the same runs."""
    path = analysis_cache_path(output)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema_version") != ANALYSIS_CACHE_SCHEMA_VERSION:
        return None
    selections = manifest.get("selections")
    encoded_outputs = manifest.get("outputs")
    if not isinstance(selections, list) or not selections:
        return None
    if not isinstance(encoded_outputs, list) or not encoded_outputs:
        return None
    outputs = [
        candidate if candidate.is_absolute() else path.parent / candidate
        for item in encoded_outputs
        if isinstance(item, str)
        for candidate in (Path(item),)
    ]
    if len(outputs) != len(encoded_outputs) or not all(
        item.is_file() for item in outputs
    ):
        return None
    previous_selections = client.analysis_selections
    client.analysis_selections = []
    try:
        for selection in selections:
            if not isinstance(selection, dict):
                return None
            try:
                grouped = completed_seed_runs(
                    client,
                    experiment_name=str(selection["experiment_name"]),
                    group_id=str(selection["group_id"]),
                    atomic_run_ids=[str(item) for item in selection["atomic_run_ids"]],
                    protocol_version=(
                        None
                        if selection.get("protocol_version") is None
                        else str(selection["protocol_version"])
                    ),
                )
            except (KeyError, TypeError):
                return None
            current_run_ids = sorted(
                run.run_id
                for runs_for_condition in grouped.values()
                for run in runs_for_condition
            )
            if current_run_ids != selection.get("run_ids"):
                return None
    finally:
        client.analysis_selections = previous_selections
    return outputs


def cached_analysis_console_output(output: Path) -> str:
    """Return console output saved with a validated analysis cache hit."""
    try:
        manifest = json.loads(analysis_cache_path(output).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    value = manifest.get("console_output", "")
    return value if isinstance(value, str) else ""


def write_analysis_cache(
    client: AnalysisClient,
    output: Path,
    outputs: Sequence[Path],
    *,
    console_output: str = "",
) -> None:
    """Persist selected run IDs and the files produced from them."""
    if not client.analysis_selections or not outputs:
        return
    path = analysis_cache_path(output)
    if not all(item.is_file() for item in outputs):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_outputs = []
    for item in outputs:
        try:
            encoded_outputs.append(str(item.relative_to(path.parent)))
        except ValueError:
            encoded_outputs.append(str(item.resolve()))
    manifest = {
        "schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
        "selections": client.analysis_selections,
        "outputs": encoded_outputs,
        "console_output": console_output,
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
