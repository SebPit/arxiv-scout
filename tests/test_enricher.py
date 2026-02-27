import json
import os
from unittest.mock import patch, MagicMock
from arxiv_scout.enricher import enrich_papers, match_affiliations, parse_s2_response
from arxiv_scout.db import Database

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "semantic_scholar_batch.json")

def test_parse_s2_response():
    with open(FIXTURE_PATH) as f:
        data = json.load(f)
    result = parse_s2_response(data[0])
    assert len(result["authors"]) == 3
    assert result["authors"][0]["name"] == "Alice Smith"
    assert result["citation_count"] == 5

def test_match_affiliations_found():
    matches = match_affiliations(["Google DeepMind", "University of Oxford"], ["Google", "DeepMind", "OpenAI"])
    assert "Google" in matches or "DeepMind" in matches

def test_match_affiliations_none():
    matches = match_affiliations(["University of Nowhere"], ["Google", "OpenAI"])
    assert len(matches) == 0

def test_match_affiliations_case_insensitive():
    matches = match_affiliations(["google research"], ["Google"])
    assert "Google" in matches

def test_enrich_papers_stores_data(tmp_db):
    db = Database(tmp_db)
    db.insert_paper(arxiv_id="2602.12345", title="A", abstract="", categories="", published_date="", arxiv_url="")
    db.insert_paper(arxiv_id="2602.67890", title="B", abstract="", categories="", published_date="", arxiv_url="")

    with open(FIXTURE_PATH) as f:
        fixture = json.load(f)

    with patch("arxiv_scout.enricher.requests.post") as mock_post, \
         patch("arxiv_scout.enricher.time.sleep"):
        mock_resp = MagicMock()
        mock_resp.json.return_value = fixture
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        enrich_papers(db, keywords=["Google", "MIT"])

    # First paper should be enriched
    p1 = db.get_paper_by_arxiv_id("2602.12345")
    assert p1["enrichment_found"] == 1
    authors = db.get_paper_authors(p1["id"])
    assert len(authors) == 3

    # Second paper (null from S2) should be marked as attempted but not found
    p2 = db.get_paper_by_arxiv_id("2602.67890")
    assert p2["enrichment_found"] == 0
    assert p2["enrichment_attempted_at"] is not None
