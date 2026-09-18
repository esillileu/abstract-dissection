"""Shared helpers for corpus CLI: source selection, store resolution, and bandwidth limits."""

from __future__ import annotations

import typer
from repro_io.http.download import BandwidthScheduler
from repro_io.s3 import S3ObjectStore

from f2.corpus.object_store import s3_config_from_environment

from ..sources import SOURCE_BY_KEY, SOURCES, CorpusSource

_BW_PEAK_DEFAULT = 40.0  # Mbit/s  (09:00-22:00 local)
_BW_OFFPEAK_DEFAULT = 100.0  # Mbit/s  (22:00-09:00 local)


def _selected(source: str) -> list[CorpusSource]:
    if source == "all":
        return list(SOURCES)
    if source not in SOURCE_BY_KEY:
        raise typer.BadParameter(
            f"source must be one of: {', '.join(SOURCE_BY_KEY)}, all"
        )
    return [SOURCE_BY_KEY[source]]


def _store(restricted: bool = False) -> S3ObjectStore:
    cfg = (
        s3_config_from_environment(restricted=True)
        if restricted
        else s3_config_from_environment()
    )
    return S3ObjectStore(cfg)


def _bandwidth(
    peak_mbps: float | None,
    offpeak_mbps: float | None,
) -> BandwidthScheduler:
    """Return a :class:`BandwidthScheduler` applying the given limits.

    Defaults: 40 Mbit/s peak (09-22), 100 Mbit/s off-peak (22-09).
    Either value can be overridden via CLI; ``0`` means use the default.
    """
    return BandwidthScheduler(
        peak_mbps=peak_mbps or _BW_PEAK_DEFAULT,
        offpeak_mbps=offpeak_mbps or _BW_OFFPEAK_DEFAULT,
    )


__all__ = [
    "_BW_OFFPEAK_DEFAULT",
    "_BW_PEAK_DEFAULT",
    "_bandwidth",
    "_selected",
    "_store",
]
