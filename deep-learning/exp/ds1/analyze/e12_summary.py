"""Final metrics for the extended-MLP/SimpleConvNet/DeepConvNet comparison."""

from .final_metrics import render_cross_group_summary


MODELS = (
    ("GT09", "MLP-EXT-ALL-BOOK"),
    ("GT06", "CNN-SIMPLE-BOOK"),
    ("GT07", "CNN-DEEP-BOOK"),
)


def render(client, error_style, output):
    del error_style
    return render_cross_group_summary(
        client,
        analysis_id="e12 summary",
        models=MODELS,
        output=output,
    )
