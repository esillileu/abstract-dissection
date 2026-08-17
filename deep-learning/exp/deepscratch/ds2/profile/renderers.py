"""DS2 profile-analysis renderer registry."""

from .e10.render import render as render_e10
from .e11.render import render as render_e11


RENDERERS = {
    "e10": render_e10,
    "e11": render_e11,
}


def resolve(study_id: str):
    return RENDERERS.get(study_id)


__all__ = ["RENDERERS", "resolve"]
