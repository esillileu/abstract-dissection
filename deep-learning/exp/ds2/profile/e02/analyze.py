"""Turn the synchronized e02 benchmark JSON into comparisons and a report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "exp/ds2/profile/e02/results/update.json"
DEFAULT_REPORT = ROOT / "exp/ds2/profile/e02/results/analysis.md"
DEFAULT_CSV = ROOT / "exp/ds2/profile/e02/results/comparisons.csv"
DEFAULT_EXPERIMENT_SUMMARY = ROOT / "exp/ds2/results/image/e02_summary.csv"
DEFAULT_NSYS_SUMMARY = ROOT / "exp/ds2/profile/e02/results/nsys/cuda_api_summary.csv"


def _by_condition(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(row["condition"]): row
        for row in payload["results"]  # type: ignore[index]
    }


def _comparison(
    rows: dict[str, dict[str, object]],
    baseline: str,
    candidate: str,
    question: str,
) -> dict[str, object]:
    baseline_ms = float(rows[baseline]["mean_ms_per_update"])
    candidate_ms = float(rows[candidate]["mean_ms_per_update"])
    return {
        "question": question,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_ms_per_update": baseline_ms,
        "candidate_ms_per_update": candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "reduction_ms_per_update": baseline_ms - candidate_ms,
        "reduction_percent": (baseline_ms - candidate_ms) / baseline_ms * 100,
    }


def build_comparisons(
    payload: dict[str, object],
) -> list[dict[str, object]]:
    rows = _by_condition(payload)
    specs = (
        (
            "original-cbow-ns",
            "implemented-cbow-ns",
            "CBOW NS: original → implemented",
        ),
        (
            "original-skipgram-ns",
            "implemented-skipgram-ns",
            "SkipGram NS: original → implemented",
        ),
        (
            "original-cbow-fs",
            "implemented-cbow-fs",
            "CBOW FS: original adaptation → implemented",
        ),
        (
            "original-skipgram-fs",
            "implemented-skipgram-fs",
            "SkipGram FS: original adaptation → implemented",
        ),
        (
            "original-cbow-onehot-fs",
            "implemented-cbow-onehot-fs",
            "CBOW One-hot FS: original adaptation → implemented",
        ),
        (
            "original-skipgram-onehot-fs",
            "implemented-skipgram-onehot-fs",
            "SkipGram One-hot FS: original adaptation → implemented",
        ),
        (
            "original-cbow-fs",
            "original-cbow-onehot-fs",
            "Original CBOW FS: embedding → one-hot",
        ),
        (
            "original-skipgram-fs",
            "original-skipgram-onehot-fs",
            "Original SkipGram FS: embedding → one-hot",
        ),
        (
            "implemented-cbow-fs",
            "implemented-cbow-onehot-fs",
            "Implemented CBOW FS: embedding → one-hot",
        ),
        (
            "implemented-skipgram-fs",
            "implemented-skipgram-onehot-fs",
            "Implemented SkipGram FS: embedding → one-hot",
        ),
        (
            "original-cbow-ns",
            "original-cbow-fs",
            "Original adaptation CBOW: NS → FS",
        ),
        (
            "original-skipgram-ns",
            "original-skipgram-fs",
            "Original adaptation SkipGram: NS → FS",
        ),
        (
            "implemented-cbow-ns",
            "implemented-cbow-fs",
            "Implemented CBOW: NS → FS",
        ),
        (
            "implemented-cbow-ns",
            "implemented-cbow-fused-ns",
            "Implemented CBOW NS: standard → fused",
        ),
        (
            "implemented-skipgram-ns",
            "implemented-skipgram-fused-ns",
            "Implemented SkipGram NS: standard → fused",
        ),
        (
            "implemented-skipgram-ns",
            "implemented-skipgram-fs",
            "Implemented SkipGram: NS → FS",
        ),
        (
            "original-cbow-ns",
            "original-skipgram-ns",
            "Original: CBOW → SkipGram",
        ),
        (
            "implemented-cbow-ns",
            "implemented-skipgram-ns",
            "Implemented NS: CBOW → SkipGram",
        ),
    )
    return [
        _comparison(rows, baseline, candidate, question)
        for baseline, candidate, question in specs
        if baseline in rows and candidate in rows
    ]


def _phase(row: dict[str, object], name: str) -> float:
    phases = row["phase_ms_per_update"]
    assert isinstance(phases, dict)
    return float(phases.get(name, 0.0))


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _phase_percent(row: dict[str, object], name: str) -> str:
    phases = row["phase_ms_per_update"]
    assert isinstance(phases, dict)
    total = sum(float(value) for value in phases.values())
    return _percent(_phase(row, name) / total) if total else "not measured"


def _faster_or_slower(percent_reduction: float) -> str:
    direction = "faster" if percent_reduction >= 0 else "slower"
    return f"{abs(percent_reduction):.1f}% {direction}"


def load_recorded_training_times(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["series"]: float(row["mean"])
            for row in csv.DictReader(handle)
            if row["metric"] == "training_time_s" and row["mean"]
        }


def load_nsys_launch_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["condition"]: int(row["total_launch_calls"])
            for row in csv.DictReader(handle)
        }


def render_report(
    payload: dict[str, object],
    recorded_training_times: dict[str, float] | None = None,
    nsys_launch_counts: dict[str, int] | None = None,
) -> str:
    rows = _by_condition(payload)
    comparisons = build_comparisons(payload)
    comparison_by_question = {str(row["question"]): row for row in comparisons}
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)

    lines = [
        "# DS2 e02 Word2Vec profiling analysis",
        "",
        "## Measurement contract",
        "",
        f"- Device: `{metadata.get('device_name', metadata['device'])}` "
        f"(`{metadata['device']}`)",
        f"- CuPy: `{metadata['cupy_version']}`",
        "- PTB train, window 5, embedding 100, batch 100, Adam 0.001",
        "- One workload-cold update is measured before warmup.",
        "- Consecutive steady updates use CUDA event pairs and synchronize once.",
        "- Throughput windows synchronize before and after all measured updates.",
        "- Runtime estimate: cold + steady throughput × (total updates - 1).",
        "- Repeat SD extrapolates between-window steady-rate variation linearly.",
        "- Phase timings use a separate synchronized diagnostic pass.",
        "- Implemented timings include the trainer's post-update loss recomputation.",
        "",
        "## Synchronized throughput",
        "",
        "| condition | cold ms | steady event p50 / p95 ms | "
        "throughput ms/update | samples/s | "
        "s/epoch | "
        "s/total (estimated) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, row in rows.items():
        epoch_seconds = row.get("estimated_seconds_per_epoch")
        total_seconds = row.get("estimated_seconds_total")
        cold_ms = float(row.get("cold_ms_per_update", 0.0))
        event_p50 = float(
            row.get("steady_event_p50_ms_per_update", row["mean_ms_per_update"])
        )
        event_p95 = float(
            row.get("steady_event_p95_ms_per_update", row["mean_ms_per_update"])
        )
        update_stdev = float(row.get("stdev_ms_per_update", 0.0))
        epoch_stdev = float(
            row.get("estimated_repeat_stdev_seconds_per_epoch", 0.0)
        )
        total_stdev = float(
            row.get("estimated_repeat_stdev_seconds_total", 0.0)
        )
        lines.append(
            f"| `{condition}` | "
            f"{cold_ms:.3f} | {event_p50:.3f} / {event_p95:.3f} | "
            f"{float(row['mean_ms_per_update']):.3f} ± {update_stdev:.3f} | "
            f"{float(row['samples_per_second']):.1f} | "
            f"{float(epoch_seconds):.1f} ± {epoch_stdev:.1f} | "
            f"{float(total_seconds):.1f} ± {total_stdev:.1f} |"
            if epoch_seconds is not None and total_seconds is not None
            else (
                f"| `{condition}` | "
                f"{cold_ms:.3f} | {event_p50:.3f} / {event_p95:.3f} | "
                f"{float(row['mean_ms_per_update']):.3f} | "
                f"{float(row['samples_per_second']):.1f} | — | — |"
            )
        )

    lines.extend(
        [
            "",
            "_All ± values are standard deviations across repeated throughput "
            "windows; epoch and total values use the same linear extrapolation._",
            "",
            "## Direct comparisons",
            "",
            "| comparison | speedup | time reduction |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['question']} | {float(row['speedup']):.2f}× | "
            f"{float(row['reduction_ms_per_update']):.3f} ms "
            f"({float(row['reduction_percent']):.1f}%) |"
        )

    original_cbow = rows["original-cbow-ns"]
    original_skipgram = rows["original-skipgram-ns"]
    implemented_cbow = rows["implemented-cbow-ns"]
    implemented_skipgram = rows["implemented-skipgram-ns"]
    cbow_gain = comparison_by_question["CBOW NS: original → implemented"]
    skipgram_gain = comparison_by_question["SkipGram NS: original → implemented"]
    cbow_fs = comparison_by_question["Implemented CBOW: NS → FS"]
    skipgram_fs = comparison_by_question["Implemented SkipGram: NS → FS"]
    cbow_onehot_fs = comparison_by_question.get(
        "Implemented CBOW FS: embedding → one-hot"
    )
    skipgram_onehot_fs = comparison_by_question.get(
        "Implemented SkipGram FS: embedding → one-hot"
    )
    recorded_training_times = recorded_training_times or {}
    nsys_launch_counts = nsys_launch_counts or {}

    lines.extend(
        [
            "",
            "## What changed and where the time went",
            "",
            "### 1. Shared-parameter deduplication was removed",
            "",
            "The original trainer scans aliased parameter lists and accumulates "
            "duplicate gradients every update. The implementation owns only "
            "`W_in` and `W_out`, so this phase disappears.",
            "",
            f"- Original CBOW deduplication: "
            f"{_phase(original_cbow, 'deduplicate_shared_parameters'):.3f} ms/update",
            f"- Original SkipGram deduplication: "
            f"{_phase(original_skipgram, 'deduplicate_shared_parameters'):.3f} ms/update",
            "",
            "### 2. Repeated Python layer loops were converted to batched arrays",
            "",
            "CBOW gathers all ten context positions in one indexed array and "
            "reduces them together. Both SkipGram objectives keep one hidden "
            "row per center and group its ten context labels. Negative sampling "
            "evaluates grouped `(batch, contexts, 6)` candidates, while full "
            "softmax evaluates one vocabulary-logit row per center. Negative "
            "candidates are represented in one tensor rather than separate "
            "positive/negative layer objects.",
            "",
            f"- End-to-end CBOW NS gain: {float(cbow_gain['speedup']):.2f}×, "
            f"{float(cbow_gain['reduction_percent']):.1f}% less time",
            f"- End-to-end SkipGram NS gain: "
            f"{float(skipgram_gain['speedup']):.2f}×, "
            f"{float(skipgram_gain['reduction_percent']):.1f}% less time",
            "",
        ]
    )
    if nsys_launch_counts:
        original_cbow_launches = nsys_launch_counts["original-cbow-ns"]
        original_skipgram_launches = nsys_launch_counts["original-skipgram-ns"]
        implemented_cbow_launches = nsys_launch_counts["implemented-cbow-ns"]
        implemented_skipgram_launches = nsys_launch_counts["implemented-skipgram-ns"]
        lines.extend(
            [
                "The short Nsight traces use the same process structure for "
                "each condition (initialization, 5 warmups, 20 measured updates, "
                "2 phase updates). Total traced launch calls were:",
                "",
                f"- CBOW NS: {original_cbow_launches:,} → "
                f"{implemented_cbow_launches:,} "
                f"({(original_cbow_launches - implemented_cbow_launches) / original_cbow_launches * 100:.1f}% fewer)",
                f"- SkipGram NS: {original_skipgram_launches:,} → "
                f"{implemented_skipgram_launches:,} "
                f"({(original_skipgram_launches - implemented_skipgram_launches) / original_skipgram_launches * 100:.1f}% fewer)",
                "",
            ]
        )
    lines.extend(
        [
            "### 3. The implementation pays for a post-update loss pass",
            "",
            "The current trainer recomputes model forward and objective forward "
            "after every optimizer update for reporting. This is not present in "
            "the original update path and is reported separately:",
            "",
            f"- Implemented CBOW NS post-update loss: "
            f"{_phase(implemented_cbow, 'post_update_loss'):.3f} ms/update "
            f"({_phase_percent(implemented_cbow, 'post_update_loss')})",
            f"- Implemented SkipGram NS post-update loss: "
            f"{_phase(implemented_skipgram, 'post_update_loss'):.3f} ms/update "
            f"({_phase_percent(implemented_skipgram, 'post_update_loss')})",
            "",
            "## Why Full Softmax can be faster than Negative Sampling",
            "",
            "Full Softmax does more arithmetic, but its dominant operations are "
            "large dense matrix multiplications (`hidden @ W_out.T` and their "
            "backward GEMMs). Those map efficiently to cuBLAS and expose enough "
            "parallel work to fill the GPU.",
            "",
            "Negative Sampling evaluates only six candidates, but it adds "
            "conditional-CDF random sampling, `searchsorted`, irregular indexed "
            "gathers, and `add.at` scatter accumulation. With batch 100 these "
            "are small, launch-heavy, and memory-irregular kernels. Reduced "
            "FLOPs therefore do not guarantee reduced wall time.",
            "",
            f"- Implemented CBOW FS versus NS: {float(cbow_fs['speedup']):.2f}× "
            f"candidate speedup, {float(cbow_fs['reduction_percent']):.1f}% "
            "time reduction (negative means FS is slower).",
            f"- Implemented SkipGram FS versus NS: "
            f"{float(skipgram_fs['speedup']):.2f}× candidate speedup, "
            f"{float(skipgram_fs['reduction_percent']):.1f}% time reduction "
            "(negative means FS is slower).",
            "",
        ]
    )
    if cbow_onehot_fs is not None and skipgram_onehot_fs is not None:
        lines.extend(
            [
                "## One-hot input projection cost",
                "",
                "Embedding FS and one-hot FS share the same full-vocabulary "
                "output logits, softmax objective, optimizer, batch, and "
                "reporting pass. Their measured difference isolates the dense "
                "one-hot construction and `one_hot @ W_in` projection path.",
                "",
                f"- Implemented CBOW one-hot FS versus embedding FS: "
                f"{float(cbow_onehot_fs['speedup']):.2f}× candidate speedup, "
                f"{float(cbow_onehot_fs['reduction_percent']):.1f}% time "
                "reduction (negative means one-hot is slower).",
                f"- Implemented SkipGram one-hot FS versus embedding FS: "
                f"{float(skipgram_onehot_fs['speedup']):.2f}× candidate "
                f"speedup, "
                f"{float(skipgram_onehot_fs['reduction_percent']):.1f}% time "
                "reduction (negative means one-hot is slower).",
                "",
            ]
        )
        onehot_trace_conditions = {
            "original-cbow-onehot-fs",
            "implemented-cbow-onehot-fs",
            "original-skipgram-onehot-fs",
            "implemented-skipgram-onehot-fs",
        }
        if onehot_trace_conditions <= nsys_launch_counts.keys():
            lines.extend(
                [
                    "- One-hot FS trace launch calls, original → implemented:",
                    f"  CBOW "
                    f"{nsys_launch_counts['original-cbow-onehot-fs']:,} → "
                    f"{nsys_launch_counts['implemented-cbow-onehot-fs']:,}; "
                    f"SkipGram "
                    f"{nsys_launch_counts['original-skipgram-onehot-fs']:,} → "
                    f"{nsys_launch_counts['implemented-skipgram-onehot-fs']:,}.",
                    "",
                ]
            )
    if nsys_launch_counts:
        cbow_ns_launches = nsys_launch_counts["implemented-cbow-ns"]
        cbow_fs_launches = nsys_launch_counts["implemented-cbow-fs"]
        skipgram_ns_launches = nsys_launch_counts["implemented-skipgram-ns"]
        skipgram_fs_launches = nsys_launch_counts["implemented-skipgram-fs"]
        lines.extend(
            [
                f"- CBOW trace launch calls: NS {cbow_ns_launches:,}, FS "
                f"{cbow_fs_launches:,} "
                f"({(cbow_ns_launches - cbow_fs_launches) / cbow_ns_launches * 100:.1f}% fewer for FS).",
                f"- SkipGram trace launch calls: NS {skipgram_ns_launches:,}, FS "
                f"{skipgram_fs_launches:,} "
                f"({(skipgram_ns_launches - skipgram_fs_launches) / skipgram_ns_launches * 100:.1f}% fewer for FS).",
                "",
            ]
        )
    lines.extend(
        [
            "All SkipGram variants keep 100 center rows and ten grouped context "
            "labels per row. NS evaluates six sampled candidates per context; "
            "both FS variants evaluate one full-vocabulary logits row per "
            "center. Embedding FS uses integer center IDs, while one-hot FS "
            "materializes dense center and grouped-target tensors.",
            "",
            "## Why the existing result can show CBOW FS as faster",
            "",
        ]
    )
    if recorded_training_times:
        recorded_cbow_ns = recorded_training_times["W2V-PTB-CBOW-NS"]
        recorded_cbow_fs = recorded_training_times["W2V-PTB-CBOW-FULL"]
        recorded_skipgram_ns = recorded_training_times["W2V-PTB-SKIPGRAM-NS"]
        recorded_skipgram_fs = recorded_training_times["W2V-PTB-SKIPGRAM-FULL"]
        lines.extend(
            [
                "The current e02 summary and this synchronized benchmark measure "
                "different boundaries:",
                "",
                f"- Existing CBOW summary: NS {recorded_cbow_ns:.1f}s, FS "
                f"{recorded_cbow_fs:.1f}s; FS appears "
                f"{(recorded_cbow_ns - recorded_cbow_fs) / recorded_cbow_ns * 100:.1f}% faster.",
                f"- Synchronized CBOW profile: NS "
                f"{float(implemented_cbow['mean_ms_per_update']):.3f} ms/update, "
                f"FS {float(rows['implemented-cbow-fs']['mean_ms_per_update']):.3f} "
                f"ms/update; FS is "
                f"{_faster_or_slower(float(cbow_fs['reduction_percent']))}.",
                f"- Existing SkipGram summary: NS {recorded_skipgram_ns:.1f}s, "
                f"FS {recorded_skipgram_fs:.1f}s; here FS is already slower.",
                "",
            ]
        )
    lines.extend(
        [
            "The catalog has `profiling.device_timing: false`. Its training "
            "windows record host wall time without waiting for the CUDA stream. "
            "A dense FS update can enqueue a few cuBLAS operations quickly, while "
            "the GPU continues working after the host timing window closes. NS "
            "has more host-visible sampling and irregular-kernel launch work.",
            "",
            "Therefore the existing CBOW number is primarily an asynchronous "
            "enqueue-time comparison, not proof that FS completes GPU work "
            "faster. Use synchronized throughput or enable device timing for "
            "runtime conclusions.",
            "",
            "## Per-model explanation",
            "",
            f"- CBOW: vectorized ten-context gather/mean, batched six-candidate "
            f"scoring, and removal of per-update alias deduplication reduce NS "
            f"time by {float(cbow_gain['reduction_ms_per_update']):.3f} ms/update "
            f"({float(cbow_gain['reduction_percent']):.1f}%).",
            f"- SkipGram: the original invokes ten independent "
            f"`NegativeSamplingLoss` objects; the adapter groups all ten "
            f"contexts under each center in one call. Together with batched "
            f"candidates and no alias scan this reduces time by "
            f"{float(skipgram_gain['reduction_ms_per_update']):.3f} ms/update "
            f"({float(skipgram_gain['reduction_percent']):.1f}%).",
            f"- After vectorization, implemented SkipGram NS is only "
            f"{_faster_or_slower((1 - float(implemented_skipgram['mean_ms_per_update']) / float(implemented_cbow['mean_ms_per_update'])) * 100)} "
            f"than implemented CBOW NS despite representing ten times "
            f"as many prediction terms per source batch.",
            "",
            "## Interpretation limits",
            "",
            "- Synchronized phase boundaries perturb very small kernels; use "
            "phase shares diagnostically, and synchronized throughput for speedups.",
            "- Nsight Systems traces are needed for exact CUDA kernel counts, "
            "launch gaps, and GEMM/scatter attribution.",
            "- Objective values across NS and FS are not directly comparable; "
            "this report compares runtime, not loss scale or embedding quality.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--experiment-summary",
        type=Path,
        default=DEFAULT_EXPERIMENT_SUMMARY,
    )
    parser.add_argument(
        "--nsys-summary",
        type=Path,
        default=DEFAULT_NSYS_SUMMARY,
    )
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    comparisons = build_comparisons(payload)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(
            payload,
            load_recorded_training_times(args.experiment_summary),
            load_nsys_launch_counts(args.nsys_summary),
        ),
        encoding="utf-8",
    )
    print(f"saved: {args.csv}")
    print(f"saved: {args.report}")


if __name__ == "__main__":
    main()
