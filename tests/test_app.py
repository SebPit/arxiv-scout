import pytest
from arxiv_scout.app import create_app
from arxiv_scout.db import Database

@pytest.fixture
def app_with_data(tmp_db):
    db = Database(tmp_db)
    pid = db.insert_paper(
        arxiv_id="2602.12345", title="Test Paper", abstract="An interesting abstract about ML.",
        categories="hep-ph", published_date="2026-02-27",
        arxiv_url="https://arxiv.org/abs/2602.12345"
    )
    db.update_heuristic_score(pid, 7.0)
    db.update_llm_score(pid, 8.0, "Novel approach to particle physics.")
    db.update_combined_score(pid, 7.6)
    aid = db.upsert_author(name="Alice Smith", semantic_scholar_id="111", h_index=50, citation_count=10000)
    db.link_paper_author(pid, aid, 0)
    db.insert_affiliation(aid, "Google DeepMind", "Google")

    app = create_app(db_path=tmp_db)
    app.config["TESTING"] = True
    return app

@pytest.fixture
def client(app_with_data):
    return app_with_data.test_client()

def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200

def test_index_contains_paper(client):
    resp = client.get("/")
    assert b"Test Paper" in resp.data

def test_filter_by_category(client):
    resp = client.get("/?category=hep-ph")
    assert b"Test Paper" in resp.data
    resp = client.get("/?category=cs.LG")
    assert b"Test Paper" not in resp.data

def test_filter_by_min_score(client):
    resp = client.get("/?min_score=9")
    assert b"Test Paper" not in resp.data

def test_api_papers_json(client):
    resp = client.get("/api/papers")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["papers"]) == 1
    assert data["papers"][0]["title"] == "Test Paper"
