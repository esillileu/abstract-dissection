"""Tests for ARC parsing, Trafilatura extraction, news classifier, language filter, validity filter, and word counter."""

from __future__ import annotations

from f2.corpus.pipeline import (
    ARCParser,
    LanguageFilter,
    NewsClassifier,
    PipelineRunner,
    TextExtractor,
    ValidityFilter,
    WordCounter,
)


def test_arc_parser_and_text_extractor(sample_arc_bytes: bytes):
    arc_record = ARCParser.parse_arc_bytes(sample_arc_bytes)
    assert arc_record is not None
    assert arc_record.http_status == 200
    assert "Global Markets Rally" in arc_record.html_body

    clean_text = TextExtractor.extract_text(arc_record.html_body, url=arc_record.url)
    assert clean_text is not None
    assert "Global Markets Rally" in clean_text
    assert "benchmark indices rose" in clean_text


def test_news_classifier_evaluation(sample_arc_bytes: bytes):
    arc_record = ARCParser.parse_arc_bytes(sample_arc_bytes)
    assert arc_record is not None
    clean_text = TextExtractor.extract_text(arc_record.html_body, url=arc_record.url)
    assert clean_text is not None

    score, is_news, details = NewsClassifier.evaluate(
        clean_text, arc_record.html_body, arc_record.url
    )
    assert is_news is True
    assert score >= 1.5
    assert details["has_dateline"] is True
    assert details["has_byline"] is True


def test_news_classifier_negative_cases():
    forum_text = "Joined: Jan 2010. Posts: 142. Quote reply: I disagree with this solution. Check my signature below."
    score, is_news, details = NewsClassifier.evaluate(
        forum_text, forum_text, "http://forum.example.com/topic/1"
    )
    assert is_news is False
    assert details["is_forum"] is True


def test_language_and_validity_filters(sample_arc_bytes: bytes):
    arc_record = ARCParser.parse_arc_bytes(sample_arc_bytes)
    assert arc_record is not None
    clean_text = TextExtractor.extract_text(arc_record.html_body, url=arc_record.url)
    assert clean_text is not None

    is_en, conf, lang = LanguageFilter.is_english(clean_text)
    assert is_en is True
    assert lang == "en"

    is_valid, reject_reason = ValidityFilter.check_validity(clean_text, min_words=50)
    assert is_valid is True
    assert reject_reason is None


def test_word_counter(sample_arc_bytes: bytes):
    arc_record = ARCParser.parse_arc_bytes(sample_arc_bytes)
    assert arc_record is not None
    clean_text = TextExtractor.extract_text(arc_record.html_body, url=arc_record.url)
    assert clean_text is not None

    words = WordCounter.count_words(clean_text)
    assert words > 50


def test_pipeline_runner_end_to_end(sample_arc_bytes: bytes):
    runner = PipelineRunner(min_words=50)
    result = runner.process(
        record_id="rec123",
        crawl_id="CC-MAIN-2012",
        url="http://www.reuters.com/article/1",
        raw_arc_compressed=sample_arc_bytes,
        inclusion_probability=0.01,
        design_weight=100.0,
        downloaded_bytes=len(sample_arc_bytes),
    )
    assert result.fetch_status == "success"
    assert result.extraction_success is True
    assert result.is_news_predicted is True
    assert result.is_english is True
    assert result.is_valid is True
    assert result.proxy_words == result.word_count
    assert result.proxy_words > 50
