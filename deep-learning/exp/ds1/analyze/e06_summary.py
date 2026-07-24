"""DS1 E06 summary: print SimpleCNN final accuracy and training wall time."""

from .final_metrics import render_summary


ATOMIC_RUN_IDS = ["CNN-SIMPLE-BOOK"]


def render(client, error_style, output):
    del error_style
    return render_summary(
        client,
        analysis_id="e06 summary",
        group_id="GT06",
        atomic_run_ids=ATOMIC_RUN_IDS,
        output=output,
    )
