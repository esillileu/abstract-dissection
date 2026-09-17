"""Stratum identification and mapping utilities."""

from __future__ import annotations


def get_stratum_id(
    crawl_id: str, prefilter_status: str, is_news_predicted: bool | int
) -> str:
    """Map record attributes to one of the 8 canonical design strata (S1 to S8)."""
    is_09_10 = "2009-2010" in str(crawl_id)
    is_pass = str(prefilter_status).lower() == "pass"
    is_pos = int(is_news_predicted) == 1

    if is_09_10:
        if is_pass:
            return "S1" if is_pos else "S2"
        else:
            return "S3" if is_pos else "S4"
    else:
        if is_pass:
            return "S5" if is_pos else "S6"
        else:
            return "S7" if is_pos else "S8"
