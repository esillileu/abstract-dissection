"""DS1 E08 summary: print every spatial-layout condition's final metrics."""

from .final_metrics import render_summary


ATOMIC_RUN_IDS = [
    "NN-MATCHED",
    "NN-MATCHED-PERMUTED",
    "CNN-SIMPLE-SPATIAL",
    "CNN-SIMPLE-SPATIAL-PERMUTED",
]


def render(client, error_style, output):
    del error_style
    return render_summary(
        client,
        analysis_id="e08 summary",
        group_id="GT08",
        atomic_run_ids=ATOMIC_RUN_IDS,
        output=output,
    )
