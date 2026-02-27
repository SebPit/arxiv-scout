"""End-to-end integration tests with mocked external APIs."""

import json
import os
from unittest.mock import patch, MagicMock

from arxiv_scout.db import Database
from arxiv_scout.pipeline import run_pipeline

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
categories:
  - hep-ph

affiliation_keywords:
  - Google
  - MIT

scoring:
  heuristic_weight: 0.4
  llm_weight: 0.6
  min_heuristic_score_for_llm: 3
  max_papers_per_day: 100

anthropic:
  model: claude-haiku-4-5-20251001

server:
  port: 5000
  host: 127.0.0.1
""")
    return str(cfg)


def test_full_pipeline_end_to_end(tmp_db, tmp_path):
    """Full pipeline: fetch -> enrich -> score with all mocked APIs."""
    config_path = _make_config(tmp_path)

    with open(os.path.join(FIXTURES, "sample_rss.xml")) as f:
        rss_xml = f.read()
    with open(os.path.join(FIXTURES, "semantic_scholar_batch.json")) as f:
        s2_data = json.load(f)

    mock_rss_resp = MagicMock()
    mock_rss_resp.text = rss_xml
    mock_rss_resp.status_code = 200
    mock_rss_resp.raise_for_status = MagicMock()

    mock_s2_resp = MagicMock()
    mock_s2_resp.json.return_value = s2_data
    mock_s2_resp.status_code = 200
    mock_s2_resp.raise_for_status = MagicMock()

    mock_llm_msg = MagicMock()
    mock_llm_msg.content = [MagicMock(text='{"score": 7, "summary": "Novel approach to particle physics using ML."}')]
    mock_anthropic_client = MagicMock()
    mock_anthropic_client.messages.create.return_value = mock_llm_msg

    with patch("arxiv_scout.fetcher.requests.get", return_value=mock_rss_resp), \
         patch("arxiv_scout.fetcher.time.sleep"), \
         patch("arxiv_scout.enricher.requests.post", return_value=mock_s2_resp), \
         patch("arxiv_scout.enricher.time.sleep"), \
         patch("arxiv_scout.scorer.anthropic.Anthropic", return_value=mock_anthropic_client):
        run_pipeline(config_path, db_path=tmp_db)

    db = Database(tmp_db)

    # Papers should be in the database
    p1 = db.get_paper_by_arxiv_id("2602.12345")
    assert p1 is not None
    assert p1["title"] == "A Novel Approach to Particle Physics"
    assert p1["heuristic_score"] is not None
    assert p1["combined_score"] is not None

    p2 = db.get_paper_by_arxiv_id("2602.67890")
    assert p2 is not None

    # Dashboard query should return scored papers
    papers = db.get_papers(min_score=0, limit=10)
    assert len(papers) >= 1


def test_pipeline_idempotent(tmp_db, tmp_path):
    """Running pipeline twice doesn't create duplicate papers."""
    config_path = _make_config(tmp_path)

    with open(os.path.join(FIXTURES, "sample_rss.xml")) as f:
        rss_xml = f.read()

    mock_rss_resp = MagicMock()
    mock_rss_resp.text = rss_xml
    mock_rss_resp.status_code = 200
    mock_rss_resp.raise_for_status = MagicMock()

    mock_s2_resp = MagicMock()
    mock_s2_resp.json.return_value = [None, None]
    mock_s2_resp.status_code = 200
    mock_s2_resp.raise_for_status = MagicMock()

    with patch("arxiv_scout.fetcher.requests.get", return_value=mock_rss_resp), \
         patch("arxiv_scout.fetcher.time.sleep"), \
         patch("arxiv_scout.enricher.requests.post", return_value=mock_s2_resp), \
         patch("arxiv_scout.enricher.time.sleep"), \
         patch("arxiv_scout.scorer.anthropic.Anthropic", side_effect=Exception("no key")):
        run_pipeline(config_path, db_path=tmp_db)
        run_pipeline(config_path, db_path=tmp_db)

    db = Database(tmp_db)
    p1 = db.get_paper_by_arxiv_id("2602.12345")
    assert p1 is not None
    # Only one row per arxiv_id
    cursor = db.execute("SELECT COUNT(*) as cnt FROM papers")
    count = cursor.fetchone()["cnt"]
    assert count == 2  # exactly 2 papers, no duplicates


def test_dashboard_shows_pipeline_results(tmp_db, tmp_path):
    """Verify the Flask dashboard renders pipeline results."""
    from arxiv_scout.app import create_app

    config_path = _make_config(tmp_path)

    with open(os.path.join(FIXTURES, "sample_rss.xml")) as f:
        rss_xml = f.read()
    with open(os.path.join(FIXTURES, "semantic_scholar_batch.json")) as f:
        s2_data = json.load(f)

    mock_rss = MagicMock(text=rss_xml, status_code=200)
    mock_rss.raise_for_status = MagicMock()
    mock_s2 = MagicMock(status_code=200)
    mock_s2.json.return_value = s2_data
    mock_s2.raise_for_status = MagicMock()

    mock_llm = MagicMock()
    mock_llm.content = [MagicMock(text='{"score": 9, "summary": "Groundbreaking work."}')]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_llm

    with patch("arxiv_scout.fetcher.requests.get", return_value=mock_rss), \
         patch("arxiv_scout.fetcher.time.sleep"), \
         patch("arxiv_scout.enricher.requests.post", return_value=mock_s2), \
         patch("arxiv_scout.enricher.time.sleep"), \
         patch("arxiv_scout.scorer.anthropic.Anthropic", return_value=mock_client):
        run_pipeline(config_path, db_path=tmp_db)

    app = create_app(db_path=tmp_db)
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/")
    assert resp.status_code == 200
    assert b"A Novel Approach to Particle Physics" in resp.data

    resp_api = client.get("/api/papers")
    data = resp_api.get_json()
    assert data["count"] >= 1
