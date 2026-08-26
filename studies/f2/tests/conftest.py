"""Pytest configuration and shared fixtures for F2 study tests."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_arc_bytes() -> bytes:
    """Constructs a valid compressed ARC record containing standard news HTML."""
    html_content = (
        "<html><head><title>Global Markets Rally as Economic Data Beats Estimates</title></head>"
        "<body>"
        "<article>"
        "<h1>Global Markets Rally as Economic Data Beats Estimates</h1>"
        "<p class='byline'>By John Smith, Reuters</p>"
        "<p class='date'>March 15, 2012</p>"
        "<p>NEW YORK — Global equity markets rallied sharply on Thursday following stronger-than-expected manufacturing data from major economies. "
        "The benchmark indices rose across Europe and North America as investor sentiment improved dramatically.</p>"
        '<p>"We are seeing robust demand across multiple sectors, which signals resilience in the underlying economy," said Jane Doe, chief market strategist at Global Investments. '
        "She added that corporate earnings have consistently outperformed analyst forecasts over the past two quarters.</p>"
        "<p>Central bank officials announced that interest rate policies would remain accommodative to support sustained long-term growth. "
        "According to the latest quarterly report, inflation pressures remained well within target thresholds, giving policymakers flexibility.</p>"
        "<p>Trading volumes were significantly higher than seasonal averages, with technology and financial shares leading the broad market advance. "
        "Analysts reported that sovereign bond yields stabilized following the economic release, reflecting reduced market volatility and growing optimism among institutional investors worldwide.</p>"
        "</article>"
        "</body></html>"
    )
    http_payload = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(html_content)}\r\n\r\n"
        f"{html_content}"
    ).encode()

    arc_header = f"http://www.reuters.com/article/2012/03/15/us-markets-idUSBRE82E0NV 192.168.1.1 20120315120000 text/html {len(http_payload)}\n".encode()
    raw_arc = arc_header + http_payload
    return gzip.compress(raw_arc)


@pytest.fixture
def sample_cluster_idx_text() -> str:
    return (
        "com,apple)/ 20120101000000\tcdx-00000.gz\t0\t200000\t1\n"
        "com,bbc)/ 20120101000000\tcdx-00000.gz\t200000\t200000\t2\n"
        "com,nytimes)/ 20120101000000\tcdx-00001.gz\t0\t200000\t3\n"
        "com,reuters)/ 20120101000000\tcdx-00001.gz\t200000\t200000\t4\n"
        "com,yahoo)/ 20120101000000\tcdx-00002.gz\t0\t200000\t5\n"
    )
