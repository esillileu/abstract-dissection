"""Pre-fetch filter rule definitions, stratum mapping, and rule ablation analysis."""

from __future__ import annotations

import re
from typing import Any

from .models import RuleAblationResult

NON_PDF_MEDIA_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|css|js|mp3|mp4|avi|zip|gz|tar|tgz|exe|dmg|iso|bin|doc|docx|ppt|pptx|xls|xlsx|rss|xml|json|swf|ico|woff|ttf|svg)(\?.*)?$",
    re.IGNORECASE,
)
PDF_EXT = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)
ALL_BINARY_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|css|js|pdf|mp3|mp4|avi|zip|gz|tar|tgz|exe|dmg|iso|bin|doc|docx|ppt|pptx|xls|xlsx|rss|xml|json|swf|ico|woff|ttf|svg)(\?.*)?$",
    re.IGNORECASE,
)
DISQUALIFIED_PATH = re.compile(
    r"/(wp-content|wp-includes|assets|static|images|img|css|js|themes|plugins|cgi-bin|cart|checkout|signin|signup|login|register|logout|privacy-policy|terms-of-service|terms-of-use|contact-us|about-us|sitemap|robots\.txt)/",
    re.IGNORECASE,
)
NON_NEWS_PATTERNS = re.compile(
    r"/(product|products|shop|store|item|items|catalog|pricing|buy|order|forum|forums|thread|threads|viewtopic|board|boards|member|members|profile|profiles|user|users|tag|tags|category|categories|search|gallery|photo|photos|video|videos|game|games|casino|poker|bet|gambling|loan|mortgage|insurance)/",
    re.IGNORECASE,
)


def get_stratum_id(
    crawl_id: str,
    prefilter_status: str | None,
    is_news_predicted: bool | int,
) -> str:
    """Map record attributes to one of the 8 canonical design strata (S1 to S8)."""
    is_09_10 = "2009-2010" in str(crawl_id)
    is_pass = (
        str(prefilter_status).lower() == "pass"
        if prefilter_status is not None
        else True
    )
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


def _get_proxy_words(r: dict[str, Any]) -> float:
    if (
        int(r.get("is_news_predicted", 0)) == 1
        and int(r.get("is_english", 0)) == 1
        and int(r.get("is_valid", 0)) == 1
    ):
        return float(r.get("word_count", 0))
    return 0.0


def evaluate_rule_ablation(
    all_records: list[dict[str, Any]],
) -> list[RuleAblationResult]:
    """Perform fine-grained ablation of pre-fetch rules on full 10k population."""
    tot_bytes = sum(float(r.get("downloaded_bytes", 0)) for r in all_records)
    tot_proxy_words = sum(_get_proxy_words(r) for r in all_records)

    r1a_records = [
        r for r in all_records if NON_PDF_MEDIA_EXT.search(str(r.get("url", "")))
    ]
    r1b_records = [r for r in all_records if PDF_EXT.search(str(r.get("url", "")))]
    r1_all_binary = [
        r for r in all_records if ALL_BINARY_EXT.search(str(r.get("url", "")))
    ]
    r2_records = [
        r for r in all_records if DISQUALIFIED_PATH.search(str(r.get("url", "")))
    ]
    r3_records = [
        r
        for r in all_records
        if 0 < int(r.get("arc_length", r.get("downloaded_bytes", 0))) < 1200
    ]
    r4_records = [
        r for r in all_records if NON_NEWS_PATTERNS.search(str(r.get("url", "")))
    ]
    r1_2_3_combined = [
        r
        for r in all_records
        if ALL_BINARY_EXT.search(str(r.get("url", "")))
        or DISQUALIFIED_PATH.search(str(r.get("url", "")))
        or (0 < int(r.get("arc_length", r.get("downloaded_bytes", 0))) < 1200)
    ]

    def _make_res(
        name: str, desc: str, sub: list[dict[str, Any]]
    ) -> RuleAblationResult:
        n_hits = len(sub)
        b_hits = sum(float(r.get("downloaded_bytes", 0)) for r in sub)
        w_hits = sum(_get_proxy_words(r) for r in sub)
        val_news = sum(
            1
            for r in sub
            if int(r.get("is_news_predicted", 0)) == 1
            and int(r.get("is_english", 0)) == 1
            and int(r.get("is_valid", 0)) == 1
        )
        return RuleAblationResult(
            rule_name=name,
            description=desc,
            hits_count=n_hits,
            hits_pct=n_hits / len(all_records) * 100,
            bytes_saved=int(b_hits),
            bytes_saved_pct=b_hits / tot_bytes * 100 if tot_bytes > 0 else 0.0,
            proxy_words_in_reject=int(w_hits),
            proxy_words_loss_pct=w_hits / tot_proxy_words * 100
            if tot_proxy_words > 0
            else 0.0,
            sample_valid_news_count=val_news,
        )

    return [
        _make_res(
            "Rule 1a: Non-PDF Binary & Media Extensions",
            "Discards .jpg, .png, .mp4, .zip, .js, .css, .exe",
            r1a_records,
        ),
        _make_res(
            "Rule 1b: PDF Documents (.pdf Only)",
            "Discards PDF document payloads",
            r1b_records,
        ),
        _make_res(
            "Rule 1 Combined: All Binary & Media Extensions (Leading)",
            "Discards all binary extensions (1a + 1b)",
            r1_all_binary,
        ),
        _make_res(
            "Rule 2: Disqualified Static / Admin Paths",
            "Discards /wp-content/, /assets/, /images/, /login/",
            r2_records,
        ),
        _make_res(
            "Rule 3: Tiny Compressed Stubs",
            "Discards records < 1,200 bytes (404s/blank stubs)",
            r3_records,
        ),
        _make_res(
            "Rule 4: Aggressive Topic Patterns (High Risk)",
            "Discards /product/, /shop/, /forum/, /category/, /search/",
            r4_records,
        ),
        _make_res(
            "Rules 1 + 2 + 3 Combined (Extended)",
            "All binary extensions + asset paths + tiny stubs",
            r1_2_3_combined,
        ),
    ]


__all__ = [
    "ALL_BINARY_EXT",
    "DISQUALIFIED_PATH",
    "NON_NEWS_PATTERNS",
    "NON_PDF_MEDIA_EXT",
    "PDF_EXT",
    "evaluate_rule_ablation",
    "get_stratum_id",
]
