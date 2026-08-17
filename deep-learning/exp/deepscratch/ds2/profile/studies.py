"""DS2 profile-study registry."""

from exp.deepscratch.profile import ProfileStudy

from .e10.study import STUDY as UPDATE_BREAKDOWN
from .e11.study import STUDY as AXIS_SCALING


STUDIES: dict[str, ProfileStudy] = {
    "update_breakdown": UPDATE_BREAKDOWN,
    "axis_scaling": AXIS_SCALING,
}


def resolve(study_kind: str) -> ProfileStudy:
    try:
        return STUDIES[study_kind]
    except KeyError as error:
        raise ValueError(f"unknown DS2 profile study kind: {study_kind}") from error


__all__ = ["STUDIES", "resolve"]
