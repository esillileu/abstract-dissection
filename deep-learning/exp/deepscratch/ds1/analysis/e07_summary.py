"""DS1 E07 summary: compare original and ten-run final accuracies."""

from .final_metrics import render_accuracy_comparison_summary


ATOMIC_RUN_ID = "CNN-DEEP-BOOK"


def render(client, error_style, output):
    del error_style
    return render_accuracy_comparison_summary(
        client,
        analysis_id="e07 summary",
        group_id="GT07",
        atomic_run_ids=[ATOMIC_RUN_ID],
        output=output,
    )
