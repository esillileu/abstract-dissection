"""DS1 GT07: reproduce the single-CNN graph layout used by GT06."""

from .e06_simple_cnn import render_cnn

ATOMIC_RUN_ID = "CNN-DEEP-BOOK"


def render(client, error_style, output):
    del output
    return render_cnn(
        client,
        error_style,
        group_id="GT07",
        atomic_run_id=ATOMIC_RUN_ID,
    )


__all__ = ["render"]
