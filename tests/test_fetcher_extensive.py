"""Extensive fetcher tests covering edge cases."""

import os
from unittest.mock import patch, MagicMock
from arxiv_scout.fetcher import parse_arxiv_feed, fetch_papers
from arxiv_scout.db import Database


def test_parse_empty_feed():
    """Empty RSS feed returns empty list."""
    papers = parse_arxiv_feed("")
    assert papers == []


def test_parse_feed_no_entries():
    """Feed with channel but no items."""
    xml = """<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns="http://purl.org/rss/1.0/">
    <channel><title>Test</title></channel>
    </rdf:RDF>"""
    papers = parse_arxiv_feed(xml)
    assert papers == []


def test_parse_feed_malformed_link():
    """Entry with non-arxiv link is skipped."""
    xml = """<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns="http://purl.org/rss/1.0/"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
    <channel><title>Test</title></channel>
    <item rdf:about="http://example.com/no-arxiv-id">
    <title>No ArXiv Link</title>
    <link>http://example.com/no-arxiv-id</link>
    <description>Some paper.</description>
    <dc:creator>John Doe</dc:creator>
    </item>
    </rdf:RDF>"""
    papers = parse_arxiv_feed(xml)
    assert papers == []


def test_parse_feed_strips_html_from_abstract():
    """HTML tags in description are stripped."""
    xml = """<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns="http://purl.org/rss/1.0/"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
    <channel><title>Test</title></channel>
    <item rdf:about="http://arxiv.org/abs/2602.55555v1">
    <title>HTML Abstract</title>
    <link>http://arxiv.org/abs/2602.55555v1</link>
    <description>arXiv:2602.55555v1  27 Feb 2026. <p>We study <b>bold</b> effects.</p></description>
    <dc:creator>Jane Doe</dc:creator>
    </item>
    </rdf:RDF>"""
    papers = parse_arxiv_feed(xml)
    assert len(papers) == 1
    assert "<p>" not in papers[0]["abstract"]
    assert "<b>" not in papers[0]["abstract"]
    assert "bold" in papers[0]["abstract"]


def test_parse_feed_single_author():
    """Single author (no comma) is handled correctly."""
    xml = """<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns="http://purl.org/rss/1.0/"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
    <channel><title>Test</title></channel>
    <item rdf:about="http://arxiv.org/abs/2602.44444v1">
    <title>Solo Paper</title>
    <link>http://arxiv.org/abs/2602.44444v1</link>
    <description>Solo abstract.</description>
    <dc:creator>Solo Author</dc:creator>
    </item>
    </rdf:RDF>"""
    papers = parse_arxiv_feed(xml)
    assert len(papers) == 1
    assert papers[0]["authors"] == ["Solo Author"]


def test_fetch_papers_multiple_categories(tmp_db):
    """Fetching from multiple categories accumulates papers."""
    db = Database(tmp_db)
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_rss.xml")
    with open(fixture_path) as f:
        xml = f.read()

    call_count = 0

    def mock_get(url, timeout=30):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.text = xml
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        return resp

    with patch("arxiv_scout.fetcher.requests.get", side_effect=mock_get), \
         patch("arxiv_scout.fetcher.time.sleep"):
        count = fetch_papers(db, ["hep-ph", "cs.LG"])

    # Called once per category
    assert call_count == 2
    # First category inserts 2, second category deduplicates (same IDs in fixture)
    assert count == 2


def test_fetch_papers_http_error(tmp_db):
    """HTTP error from arXiv raises exception."""
    import requests as req
    db = Database(tmp_db)

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = req.HTTPError("503 Service Unavailable")

    with patch("arxiv_scout.fetcher.requests.get", return_value=mock_resp), \
         patch("arxiv_scout.fetcher.time.sleep"):
        try:
            fetch_papers(db, ["hep-ph"])
            assert False, "Should have raised"
        except req.HTTPError:
            pass
