"""Final metrics for the cross-group DeepConvNet/extended-MLP comparison."""

from .final_metrics import render_cross_group_summary


MODELS = (
    ("GT07", "CNN-DEEP-BOOK"),
    ("GT09", "MLP-EXT-ALL-BOOK"),
)


def render(client, error_style, output):
    del error_style
    return render_cross_group_summary(
        client,
        analysis_id="e12 summary",
        models=MODELS,
        output=output,
    )
