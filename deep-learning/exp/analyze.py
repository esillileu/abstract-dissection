"""Shared, seed-aware MLflow analysis helpers for experiment domains."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from exp.plot_theme import MUTED, apply_plot_theme


apply_plot_theme()


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


class AnalysisClient:
    """Delegate MLflow calls while carrying analysis-only selection state."""

    def __init__(self, client, *, seed: int | None = None) -> None:
        self._client = client
        self.analysis_seed = None if seed is None else str(seed)

    def __getattr__(self, name):
        return getattr(self._client, name)


@dataclass(frozen=True)
class RunRef:
    run_id: str
    atomic_run_id: str
    seed: str
    start_time: int
    local_artifact_root: Path | None = None


@dataclass(frozen=True)
class Curve:
    steps: np.ndarray
    mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    run_count: int

    @classmethod
    def empty(cls) -> "Curve":
        empty = np.asarray([], dtype=float)
        return cls(empty, empty, empty, empty, 0)


def mlflow_client(tracking_uri: str):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install tracking dependencies with `uv sync --extra tracking`.") from exc
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def completed_seed_runs(
    client,
    *,
    experiment_name: str,
    group_id: str,
    atomic_run_ids: Iterable[str],
) -> dict[str, list[RunRef]]:
    """Return the newest completed attempt for every (condition, seed)."""
    wanted = tuple(atomic_run_ids)
    grouped: dict[str, list[RunRef]] = {atomic: [] for atomic in wanted}
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return grouped
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' and "
            f"tags.`execution_group.id` = '{group_id}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=5_000,
    )
    selected: dict[tuple[str, str], RunRef] = {}
    for run in runs:
        tags = run.data.tags
        if tags.get("run.type") != "seed_trial":
            continue
        if tags.get("execution_group.id") != group_id:
            continue
        atomic = tags.get("atomic_run.id", "")
        if atomic not in grouped:
            continue
        seed = run.data.params.get("seed/master", run.data.params.get("seed", run.info.run_id))
        selected_seed = getattr(client, "analysis_seed", None)
        if selected_seed is not None and str(seed) != selected_seed:
            continue
        key = (atomic, str(seed))
        selected.setdefault(
            key,
            RunRef(
                run.info.run_id,
                atomic,
                str(seed),
                int(run.info.start_time or 0),
                _local_artifact_root(experiment_name, tags.get("run.key")),
            ),
        )
    for (atomic, _), run in selected.items():
        grouped[atomic].append(run)
    for runs_for_condition in grouped.values():
        runs_for_condition.sort(key=lambda run: run.seed)
    return grouped


def _local_artifact_root(experiment_name: str, run_key: str | None) -> Path | None:
    if not run_key:
        return None
    path = Path("exp") / experiment_name / "results" / "mlflow_artifacts" / run_key
    return path if path.is_dir() else None


def artifact_file(client, run: RunRef, artifact_path: str) -> Path | None:
    """Resolve the local schema-v1 mirror, then fall back to MLflow download."""
    if run.local_artifact_root is not None:
        local_path = run.local_artifact_root / artifact_path
        if local_path.is_file():
            return local_path
    try:
        downloaded = Path(client.download_artifacts(run.run_id, artifact_path))
    except Exception:
        return None
    return downloaded if downloaded.is_file() else None


def artifact_rows(client, run: RunRef, artifact_path: str) -> list[dict[str, str]]:
    """Load one CSV artifact. A missing artifact represents no history."""
    local_path = artifact_file(client, run, artifact_path)
    if local_path is None:
        return []
    with local_path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def histories_from_artifact(
    client,
    runs: Sequence[RunRef],
    *,
    artifact_path: str,
    x: str,
    y: str,
    row_filter: Callable[[Mapping[str, str]], bool] | None = None,
    x_value: Callable[[Mapping[str, str]], float] | None = None,
    y_value: Callable[[Mapping[str, str]], float] | None = None,
) -> list[dict[float, float]]:
    histories: list[dict[float, float]] = []
    for run in runs:
        history: dict[float, float] = {}
        for row in artifact_rows(client, run, artifact_path):
            if row_filter is not None and not row_filter(row):
                continue
            try:
                step = x_value(row) if x_value is not None else float(row[x])
                value = y_value(row) if y_value is not None else float(row[y])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(step) and np.isfinite(value):
                history[float(step)] = float(value)
        if history:
            histories.append(history)
    return histories


def metric_histories(client, runs: Sequence[RunRef], metric: str) -> list[dict[float, float]]:
    histories = []
    for run in runs:
        values = {
            float(item.step): float(item.value)
            for item in client.get_metric_history(run.run_id, metric)
            if np.isfinite(item.value)
        }
        if values:
            histories.append(values)
    return histories


def aggregate(histories: Sequence[Mapping[float, float]]) -> Curve:
    """Aggregate only x coordinates shared by every available seed history."""
    if not histories:
        return Curve.empty()
    common_steps = set(histories[0])
    for history in histories[1:]:
        common_steps.intersection_update(history)
    if not common_steps:
        return Curve.empty()
    steps = np.asarray(sorted(common_steps), dtype=float)
    values = np.asarray([[history[step] for step in steps] for history in histories], dtype=float)
    return Curve(
        steps=steps,
        mean=values.mean(axis=0),
        minimum=values.min(axis=0),
        maximum=values.max(axis=0),
        run_count=len(histories),
    )


def smooth_book(values: Sequence[float]) -> np.ndarray:
    """Reproduce the book's 11-point Kaiser smoothing."""
    array = np.asarray(values, dtype=float)
    window_len = 11
    if len(array) < window_len:
        return array
    reflected = np.r_[array[window_len - 1 : 0 : -1], array, array[-1 : -window_len : -1]]
    window = np.kaiser(window_len, 2)
    smoothed = np.convolve(window / window.sum(), reflected, mode="valid")
    return smoothed[5 : len(smoothed) - 5]


def smooth_histories(histories: Sequence[Mapping[float, float]]) -> list[dict[float, float]]:
    output = []
    for history in histories:
        steps = sorted(history)
        values = smooth_book([history[step] for step in steps])
        output.append({step: float(value) for step, value in zip(steps, values, strict=True)})
    return output


def plot_curve(
    axis,
    curve: Curve,
    *,
    label: str,
    error_style: str,
    marker: str | None = None,
    linestyle: str = "-",
    error_every: int = 5,
    color: str | None = None,
):
    if not len(curve.steps):
        return None
    line = axis.plot(
        curve.steps,
        curve.mean,
        label=f"{label} (n={curve.run_count})",
        marker=marker,
        markevery=max(1, error_every),
        markersize=5,
        linestyle=linestyle,
        linewidth=1.6,
        color=color,
    )[0]
    if error_style == "band":
        axis.fill_between(
            curve.steps,
            curve.minimum,
            curve.maximum,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )
    elif error_style == "errorbar":
        errors = np.maximum(
            0.0,
            np.vstack((curve.mean - curve.minimum, curve.maximum - curve.mean)),
        )
        axis.errorbar(
            curve.steps,
            curve.mean,
            yerr=errors,
            fmt="none",
            ecolor=line.get_color(),
            errorevery=max(1, error_every),
            elinewidth=0.8,
            capsize=1.5,
        )
    else:
        raise ValueError(f"unknown error style: {error_style}")
    return line


def mark_empty(axis, message: str = "No completed runs") -> None:
    if axis.has_data():
        return
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, color=MUTED)


def save_figure(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    match_original = getattr(figure, "_analysis_match_original_canvas", False)
    if (
        not match_original
        and not getattr(figure, "_analysis_skip_tight_layout", False)
    ):
        figure.tight_layout()
    if match_original:
        figure.savefig(path, dpi=figure.dpi)
    else:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    return path


def write_summary(path: Path, curves: Mapping[str, Curve]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["series", "seed_runs", "points", "final_mean", "final_min", "final_max"],
        )
        writer.writeheader()
        for name, curve in curves.items():
            writer.writerow(
                {
                    "series": name,
                    "seed_runs": curve.run_count,
                    "points": len(curve.steps),
                    "final_mean": curve.mean[-1] if len(curve.mean) else "",
                    "final_min": curve.minimum[-1] if len(curve.minimum) else "",
                    "final_max": curve.maximum[-1] if len(curve.maximum) else "",
                }
            )
    return path


def tracking_uri_default() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def parse_experiment_selection(
    values: Sequence[str], available: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Expand 01, e01, 01-08, and comma-separated analysis selections."""
    supported = tuple(available)
    if not values or any(value.lower() == "all" for value in values):
        return list(supported), []
    requested: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip().lower()
            match = re.fullmatch(r"e?(\d+)(?:-e?(\d+))?", item)
            if match is None:
                raise ValueError(f"invalid experiment selection: {item}")
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
            if start > end:
                raise ValueError(f"experiment range must be ascending: {item}")
            requested.extend(f"e{number:02d}" for number in range(start, end + 1))
    unique = list(dict.fromkeys(requested))
    return [item for item in unique if item in supported], [item for item in unique if item not in supported]
