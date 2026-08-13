"""DS1 E06 summary: compare original and ten-run final accuracies."""

from .final_metrics import render_accuracy_comparison_summary


ATOMIC_RUN_IDS = ["CNN-SIMPLE-BOOK"]


def render(client, error_style, output):
    del error_style
    return render_accuracy_comparison_summary(
        client,
        analysis_id="e06 summary",
        group_id="GT06",
        atomic_run_ids=ATOMIC_RUN_IDS,
        output=output,
    )
