from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from repro_io.http.download import BandwidthScheduler

from f2.corpus.canonical import (
    DeterministicSharder,
    extract_gigaword_documents,
    extract_wikipedia_records,
    iter_tar_records,
    normalize_text,
)
from f2.corpus.sources import SOURCE_BY_KEY, SOURCES, VALIDATION_PROFILES


def _tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_frozen_source_order_and_wmt_years() -> None:
    assert [source.key for source in SOURCES] == [
        "lm1b",
        "wmt",
        "gigaword",
        "umbc",
        "wikipedia",
    ]
    assert [item.year for item in SOURCE_BY_KEY["wmt"].files] == list(range(2007, 2013))
    assert SOURCE_BY_KEY["gigaword"].blocked_reason == "authorized LDC access required"
    assert (
        VALIDATION_PROFILES["word2vec-2013-compatibility-v1"]["maximum_verdict"]
        == "compatible_reconstruction"
    )


def test_lm1b_excludes_heldout_and_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "lm.tgz"
    _tar(
        archive,
        {
            "x/training-monolingual.tokenized.shuffled/news.en-00001-of-00100": b"train\n",
            "x/heldout-monolingual.tokenized.shuffled/news.en.heldout": b"heldout\n",
        },
    )
    assert [payload for _, payload in iter_tar_records(archive, "lm1b")] == [b"train\n"]

    unsafe = tmp_path / "unsafe.tgz"
    _tar(unsafe, {"../training-monolingual.tokenized.shuffled/escape": b"bad"})
    with pytest.raises(ValueError, match="unsafe archive member"):
        list(iter_tar_records(unsafe, "lm1b"))


def test_umbc_includes_only_plain_webbase_files(tmp_path: Path) -> None:
    archive = tmp_path / "umbc.tgz"
    _tar(
        archive,
        {
            "webbase_all/a.txt": b"plain\n",
            "webbase_all_tagged/a.txt": b"word_NN\n",
            "webbase_all/readme": b"ignore\n",
        },
    )
    assert [(name, payload) for name, payload in iter_tar_records(archive, "umbc")] == [
        ("webbase_all/a.txt", b"plain\n")
    ]


def test_normalize_text_matches_shell_recipe_fixture() -> None:
    original = "It's “Version-2”, wow! <br /> 19\n"
    assert normalize_text(original) == 'it \' s  " version -   "  , wow !      \n'


def test_normalize_text_exact_bash_equivalence() -> None:
    import subprocess

    bash_code = """normalize_text() {
  awk '{print tolower($0);}' | sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\\"/g" -e "s/”/\\"/g" \\
  -e 's/"/ " /g' -e 's/\\./ \\. /g' -e 's/<br \\/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\\!/ \\! /g' \\
  -e 's/\\?/ \\? /g' -e 's/\\;/ /g' -e 's/\\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \\
  -e 's/«/ /g' | tr 0-9 " "
}
normalize_text
"""  # noqa: RUF001
    test_cases = [
        "Hello, world! 123 test’s “quote” (parenthesis) a*b=c; d:e-f <br /> end.",  # noqa: RUF001
        "comma,no_space comma, with_space",
        "single'quote ''double'' '''triple'''",
        "Special: 1234567890 & % $ # @ ! ? ; : / \\ | = + * - _ ~ ` ^ < > [ ] { }",
        "French «guillemets» and prime 5′ and smart quotes ‘single’ “double”",  # noqa: RUF001
        "Mixed CASE WITH 99 NUMBERS and   spaces    and tabs\t",
    ]
    for tc in test_cases:
        res = subprocess.run(
            ["bash", "-c", bash_code],
            input=tc,
            text=True,
            capture_output=True,
            env={"LC_ALL": "C"},
        )
        expected = res.stdout.rstrip("\n")
        assert normalize_text(tc) == expected


def test_gigaword_and_wikipedia_fixtures() -> None:
    sgml = '<DOC id="1"><HEADLINE>Ignored</HEADLINE><TEXT><P>First &amp; second.</P><P>Third.</P></TEXT></DOC>'
    assert list(extract_gigaword_documents(sgml)) == ["First & second.\nThird."]
    xml = """<mediawiki><page><title>A</title><revision><text>Visible [[Target|label]] 2.</text></revision></page>
    <page><title>B</title><redirect title="A"/><revision><text>#REDIRECT [[A]]</text></revision></page></mediawiki>"""
    extracted = list(extract_wikipedia_records(xml))
    assert extracted == ["Visible label 2."]
    assert [normalize_text(text).strip() for text in extracted] == ["visible label   ."]


def test_shards_are_record_bounded_and_repeatable(tmp_path: Path) -> None:
    records = [
        ("a", "one two three"),
        ("b", "four five"),
        ("c", "six seven eight nine ten eleven"),
    ]
    first = DeterministicSharder(tmp_path / "a", target_words=5).write(
        records, source="fixture"
    )
    second = DeterministicSharder(tmp_path / "b", target_words=5).write(
        records, source="fixture"
    )
    assert [item.word_count for item in first] == [5, 6]
    assert [item.physical_sha256 for item in first] == [
        item.physical_sha256 for item in second
    ]
    assert (tmp_path / "a" / "manifest.json").read_bytes() == (
        tmp_path / "b" / "manifest.json"
    ).read_bytes()
    assert (
        hashlib.sha256((tmp_path / "a" / "manifest.json").read_bytes()).hexdigest()
        == hashlib.sha256((tmp_path / "b" / "manifest.json").read_bytes()).hexdigest()
    )
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["summary"]["total_shards"] == 2
    assert manifest["summary"]["total_lexical_words"] == 11
    assert manifest["summary"]["total_newlines"] == 3
    assert manifest["summary"]["total_word2vec_train_words"] == 14
    assert manifest["shards"][1]["source_span"] == {
        "source": "fixture",
        "first": "c",
        "last": "c",
    }


def test_bandwidth_scheduler_peak_and_offpeak_limits() -> None:
    current_hour = 12
    scheduler = BandwidthScheduler(
        peak_mbps=40.0,
        offpeak_mbps=100.0,
        peak_start=9,
        peak_end=22,
        current_hour_fn=lambda: current_hour,
    )

    # Peak hours: 09:00 - 21:59 -> 40 Mbps = 5,000,000 bytes/s
    for hour in [9, 12, 18, 21]:
        current_hour = hour
        assert scheduler.limit_bps() == 40.0 * 125_000.0

    # Off-peak hours: 22:00 - 08:59 -> 100 Mbps = 12,500,000 bytes/s
    for hour in [22, 23, 0, 4, 8]:
        current_hour = hour
        assert scheduler.limit_bps() == 100.0 * 125_000.0


def test_bandwidth_scheduler_drain_throttle() -> None:
    simulated_time = 100.0
    sleeps: list[float] = []

    def mock_time() -> float:
        return simulated_time

    def mock_sleep(seconds: float) -> None:
        nonlocal simulated_time
        sleeps.append(seconds)
        simulated_time += seconds

    scheduler = BandwidthScheduler(
        peak_mbps=8.0,  # 8 Mbit/s = 1,000,000 bytes/s (1 MB/s)
        offpeak_mbps=80.0,
        current_hour_fn=lambda: 12,  # peak: 1 MB/s
        time_fn=mock_time,
        sleep_fn=mock_sleep,
    )

    # Initial state: 0 tokens. Requesting 500,000 bytes.
    # Deficit = 500,000 bytes / 1,000,000 bytes/s = 0.5s wait
    scheduler.drain(500_000)
    assert len(sleeps) == 1
    assert pytest.approx(sleeps[0], 0.01) == 0.5

    # Simulate 2.0s passing without draining -> bucket fills up to capacity (1s = 1,000,000 bytes)
    simulated_time += 2.0
    sleeps.clear()

    # Requesting 400,000 bytes should not sleep because bucket has 1,000,000 tokens
    scheduler.drain(400_000)
    assert len(sleeps) == 0


def test_bandwidth_scheduler_invalid_limits() -> None:
    with pytest.raises(ValueError, match="bandwidth limits must be positive"):
        BandwidthScheduler(peak_mbps=0, offpeak_mbps=100.0)
    with pytest.raises(ValueError, match="bandwidth limits must be positive"):
        BandwidthScheduler(peak_mbps=40.0, offpeak_mbps=-10.0)
