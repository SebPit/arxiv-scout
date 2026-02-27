"""Extensive Flask app tests covering edge cases."""

import pytest
from arxiv_scout.app import create_app
from arxiv_scout.db import Database


@pytest.fixture
def seeded_db(tmp_db):
    """Create a DB with multiple papers for comprehensive testing."""
    db = Database(tmp_db)

    # Paper 1: High score, hep-ph, Google author
    pid1 = db.insert_paper(
        arxiv_id="2602.11111", title="High Score Paper",
        abstract="A groundbreaking paper about neural networks.",
        categories="hep-ph", published_date="2026-02-27",
        arxiv_url="https://arxiv.org/abs/2602.11111"
    )
    db.update_heuristic_score(pid1, 9.0)
    db.update_llm_score(pid1, 9.5, "Potentially groundbreaking.")
    db.update_combined_score(pid1, 9.3)
    aid1 = db.upsert_author(name="Alice", semantic_scholar_id="a1", h_index=80, citation_count=50000)
    db.link_paper_author(pid1, aid1, 0)
    db.insert_affiliation(aid1, "Google DeepMind", "Google")

    # Paper 2: Medium score, cs.LG
    pid2 = db.insert_paper(
        arxiv_id="2602.22222", title="Medium Score Paper",
        abstract="A solid contribution to machine learning.",
        categories="cs.LG", published_date="2026-02-26",
        arxiv_url="https://arxiv.org/abs/2602.22222"
    )
    db.update_heuristic_score(pid2, 5.0)
    db.update_llm_score(pid2, 6.0, "Solid work.")
    db.update_combined_score(pid2, 5.6)
    aid2 = db.upsert_author(name="Bob", semantic_scholar_id="b1", h_index=20, citation_count=2000)
    db.link_paper_author(pid2, aid2, 0)

    # Paper 3: Low score, hep-ph
    pid3 = db.insert_paper(
        arxiv_id="2602.33333", title="Low Score Paper",
        abstract="Incremental improvement.",
        categories="hep-ph", published_date="2026-02-25",
        arxiv_url="https://arxiv.org/abs/2602.33333"
    )
    db.update_heuristic_score(pid3, 1.0)
    db.update_combined_score(pid3, 1.0)

    return tmp_db


@pytest.fixture
def client(seeded_db):
    app = create_app(db_path=seeded_db)
    app.config["TESTING"] = True
    return app.test_client()


def test_index_default_shows_all_scored(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"High Score Paper" in resp.data


def test_filter_category_hep_ph(client):
    resp = client.get("/?category=hep-ph")
    assert b"High Score Paper" in resp.data
    assert b"Medium Score Paper" not in resp.data


def test_filter_category_cs_lg(client):
    resp = client.get("/?category=cs.LG")
    assert b"Medium Score Paper" in resp.data
    assert b"High Score Paper" not in resp.data


def test_filter_min_score_high(client):
    resp = client.get("/?min_score=8")
    assert b"High Score Paper" in resp.data
    assert b"Medium Score Paper" not in resp.data
    assert b"Low Score Paper" not in resp.data


def test_filter_min_score_zero_shows_all(client):
    resp = client.get("/?min_score=0")
    assert b"High Score Paper" in resp.data
    assert b"Low Score Paper" in resp.data


def test_sort_by_published_date(client):
    resp = client.get("/?sort=published_date&min_score=0")
    assert resp.status_code == 200
    # All papers should be present
    assert b"High Score Paper" in resp.data


def test_sort_by_invalid_column(client):
    """Invalid sort column doesn't crash (falls back to combined_score)."""
    resp = client.get("/?sort=nonexistent&min_score=0")
    assert resp.status_code == 200


def test_api_papers_returns_json(client):
    resp = client.get("/api/papers?min_score=0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "papers" in data
    assert "count" in data
    assert data["count"] == 3


def test_api_papers_category_filter(client):
    resp = client.get("/api/papers?category=cs.LG&min_score=0")
    data = resp.get_json()
    assert data["count"] == 1
    assert data["papers"][0]["title"] == "Medium Score Paper"


def test_api_papers_limit(client):
    resp = client.get("/api/papers?min_score=0&limit=1")
    data = resp.get_json()
    assert len(data["papers"]) == 1


def test_pagination_page_1(client):
    resp = client.get("/?page=1&min_score=0")
    assert resp.status_code == 200


def test_pagination_page_out_of_range(client):
    """Page beyond data returns 200 with no papers."""
    resp = client.get("/?page=999&min_score=0")
    assert resp.status_code == 200
    assert b"No papers found" in resp.data


def test_index_shows_affiliation_badge(client):
    """Google affiliation badge appears for high-score paper."""
    resp = client.get("/?min_score=0")
    assert b"Google" in resp.data


def test_index_shows_llm_summary(client):
    resp = client.get("/?min_score=0")
    assert b"Potentially groundbreaking" in resp.data


def test_empty_database(tmp_db):
    """Dashboard with no papers shows empty state."""
    app = create_app(db_path=tmp_db)
    app.config["TESTING"] = True
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"No papers found" in resp.data
