"""Extensive enricher tests covering edge cases."""

import json
from unittest.mock import patch, MagicMock
from arxiv_scout.enricher import match_affiliations, parse_s2_response, enrich_papers
from arxiv_scout.db import Database


def test_match_affiliations_multiple_keywords():
    """Multiple keywords match different affiliations."""
    affs = ["Google DeepMind", "MIT", "Stanford University"]
    kws = ["Google", "MIT", "Stanford", "OpenAI"]
    matches = match_affiliations(affs, kws)
    assert set(matches) == {"Google", "MIT", "Stanford"}


def test_match_affiliations_empty_inputs():
    assert match_affiliations([], ["Google"]) == []
    assert match_affiliations(["Google"], []) == []
    assert match_affiliations([], []) == []


def test_match_affiliations_partial_match():
    """Keyword 'Meta' matches 'Meta AI Research'."""
    matches = match_affiliations(["Meta AI Research"], ["Meta"])
    assert "Meta" in matches


def test_parse_s2_response_missing_fields():
    """S2 response with missing optional fields."""
    data = {
        "authors": [
            {"name": "Unknown", "authorId": None, "affiliations": None, "hIndex": None, "citationCount": None}
        ],
        "citationCount": None,
    }
    result = parse_s2_response(data)
    assert len(result["authors"]) == 1
    assert result["authors"][0]["affiliations"] == []
    assert result["citation_count"] is None


def test_parse_s2_response_empty_authors():
    data = {"authors": [], "citationCount": 0}
    result = parse_s2_response(data)
    assert result["authors"] == []
    assert result["citation_count"] == 0


def test_enrich_no_papers_to_enrich(tmp_db):
    """Enricher does nothing when no papers need enrichment."""
    db = Database(tmp_db)
    # Mark all papers as enriched or just have none
    with patch("arxiv_scout.enricher.requests.post") as mock_post:
        enrich_papers(db, keywords=["Google"])
        mock_post.assert_not_called()


def test_enrich_with_api_key(tmp_db):
    """API key is sent in headers."""
    db = Database(tmp_db)
    db.insert_paper(arxiv_id="2602.11111", title="A", abstract="", categories="", published_date="", arxiv_url="")

    mock_resp = MagicMock()
    mock_resp.json.return_value = [None]
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch("arxiv_scout.enricher.requests.post", return_value=mock_resp) as mock_post, \
         patch("arxiv_scout.enricher.time.sleep"):
        enrich_papers(db, keywords=["Google"], api_key="my-secret-key")

    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["headers"]["x-api-key"] == "my-secret-key"


def test_enrich_batch_size(tmp_db):
    """Papers are batched according to batch_size."""
    db = Database(tmp_db)
    for i in range(5):
        db.insert_paper(arxiv_id=f"2602.{10000+i}", title=f"P{i}", abstract="", categories="", published_date="", arxiv_url="")

    mock_resp = MagicMock()
    mock_resp.json.return_value = [None, None]
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch("arxiv_scout.enricher.requests.post", return_value=mock_resp) as mock_post, \
         patch("arxiv_scout.enricher.time.sleep"):
        enrich_papers(db, keywords=[], batch_size=2)

    # 5 papers / batch_size 2 = 3 API calls (2 + 2 + 1)
    assert mock_post.call_count == 3


def test_enrich_mixed_found_and_null(tmp_db):
    """Mix of found and null S2 responses handled correctly."""
    db = Database(tmp_db)
    db.insert_paper(arxiv_id="2602.11111", title="Found", abstract="", categories="", published_date="", arxiv_url="")
    db.insert_paper(arxiv_id="2602.22222", title="Not Found", abstract="", categories="", published_date="", arxiv_url="")

    s2_response = [
        {
            "paperId": "abc",
            "externalIds": {"ArXiv": "2602.11111"},
            "citationCount": 10,
            "authors": [
                {"authorId": "a1", "name": "Alice", "affiliations": ["Google"], "hIndex": 30, "citationCount": 5000}
            ],
        },
        None,
    ]

    mock_resp = MagicMock()
    mock_resp.json.return_value = s2_response
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch("arxiv_scout.enricher.requests.post", return_value=mock_resp), \
         patch("arxiv_scout.enricher.time.sleep"):
        enrich_papers(db, keywords=["Google"])

    p1 = db.get_paper_by_arxiv_id("2602.11111")
    assert p1["enrichment_found"] == 1
    assert p1["citation_count"] == 10

    p2 = db.get_paper_by_arxiv_id("2602.22222")
    assert p2["enrichment_found"] == 0
