import os
from unittest.mock import patch, MagicMock
from arxiv_scout.fetcher import parse_arxiv_feed, fetch_papers
from arxiv_scout.db import Database

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_rss.xml")

def test_parse_arxiv_feed():
    with open(FIXTURE_PATH) as f:
        papers = parse_arxiv_feed(f.read())
    assert len(papers) == 2
    assert papers[0]["arxiv_id"] == "2602.12345"
    assert papers[0]["title"] == "A Novel Approach to Particle Physics"
    assert "novel method" in papers[0]["abstract"]
    assert len(papers[0]["authors"]) == 3

def test_parse_extracts_clean_abstract():
    with open(FIXTURE_PATH) as f:
        papers = parse_arxiv_feed(f.read())
    # Abstract should not contain the arXiv ID prefix
    for p in papers:
        assert not p["abstract"].startswith("arXiv:")

def test_fetch_papers_stores_in_db(tmp_db):
    db = Database(tmp_db)
    with open(FIXTURE_PATH) as f:
        xml = f.read()
    with patch("arxiv_scout.fetcher.requests.get") as mock_get, \
         patch("arxiv_scout.fetcher.time.sleep"):
        mock_resp = MagicMock()
        mock_resp.text = xml
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        count = fetch_papers(db, ["hep-ph"])
    assert count == 2
    paper = db.get_paper_by_arxiv_id("2602.12345")
    assert paper is not None
    assert paper["title"] == "A Novel Approach to Particle Physics"
    authors = db.get_paper_authors(paper["id"])
    assert len(authors) == 3

def test_fetch_papers_deduplicates(tmp_db):
    db = Database(tmp_db)
    with open(FIXTURE_PATH) as f:
        xml = f.read()
    with patch("arxiv_scout.fetcher.requests.get") as mock_get, \
         patch("arxiv_scout.fetcher.time.sleep"):
        mock_resp = MagicMock()
        mock_resp.text = xml
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        count1 = fetch_papers(db, ["hep-ph"])
        count2 = fetch_papers(db, ["hep-ph"])
    assert count1 == 2
    assert count2 == 0  # all duplicates
