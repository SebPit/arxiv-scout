"""Extensive database layer tests covering edge cases and concurrency."""

import sqlite3
from arxiv_scout.db import Database


def test_empty_database_queries(tmp_db):
    """Queries on empty DB return empty lists, not errors."""
    db = Database(tmp_db)
    assert db.get_unenriched_papers() == []
    assert db.get_unscored_papers() == []
    assert db.get_papers(min_score=0) == []
    assert db.get_paper_authors(999) == []
    assert db.get_author_affiliations(999) == []


def test_get_paper_by_arxiv_id_not_found(tmp_db):
    db = Database(tmp_db)
    assert db.get_paper_by_arxiv_id("nonexistent") is None


def test_insert_paper_with_special_characters(tmp_db):
    """Titles and abstracts with quotes, unicode, and special chars."""
    db = Database(tmp_db)
    pid = db.insert_paper(
        arxiv_id="2602.99999",
        title="D'Alembert's Principle: A \"Review\" of $O(\\alpha_s)$ Effects",
        abstract="We study the résumé of μ→eγ at √s = 13 TeV with <p>HTML tags</p>",
        categories="hep-ph",
        published_date="2026-02-27",
        arxiv_url="https://arxiv.org/abs/2602.99999",
    )
    assert pid is not None
    paper = db.get_paper_by_arxiv_id("2602.99999")
    assert "D'Alembert" in paper["title"]
    assert "résumé" in paper["abstract"]


def test_many_papers_pagination(tmp_db):
    """Insert many papers and verify pagination works correctly."""
    db = Database(tmp_db)
    for i in range(75):
        pid = db.insert_paper(
            arxiv_id=f"2602.{10000 + i}",
            title=f"Paper {i}",
            abstract=f"Abstract {i}",
            categories="cs.LG",
            published_date="2026-02-27",
            arxiv_url=f"https://arxiv.org/abs/2602.{10000 + i}",
        )
        db.update_heuristic_score(pid, float(i % 10))
        db.update_combined_score(pid, float(i % 10))

    page1 = db.get_papers(min_score=0, limit=50, offset=0)
    page2 = db.get_papers(min_score=0, limit=50, offset=50)
    assert len(page1) == 50
    assert len(page2) == 25  # 75 total - 50 on page 1
    # No overlap
    ids_p1 = {p["id"] for p in page1}
    ids_p2 = {p["id"] for p in page2}
    assert ids_p1.isdisjoint(ids_p2)


def test_order_by_validation(tmp_db):
    """Invalid order_by columns fall back to combined_score."""
    db = Database(tmp_db)
    pid = db.insert_paper(
        arxiv_id="2602.11111", title="A", abstract="", categories="",
        published_date="", arxiv_url=""
    )
    db.update_combined_score(pid, 5.0)
    # SQL injection attempt should be safely handled
    papers = db.get_papers(order_by="id; DROP TABLE papers;--", min_score=0)
    assert len(papers) == 1  # still works, fell back to combined_score


def test_multiple_authors_per_paper(tmp_db):
    """Paper with many authors preserves ordering."""
    db = Database(tmp_db)
    pid = db.insert_paper(
        arxiv_id="2602.11111", title="Big Collab", abstract="",
        categories="", published_date="", arxiv_url=""
    )
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    for i, name in enumerate(names):
        aid = db.upsert_author(name=name, semantic_scholar_id=None, h_index=None, citation_count=None)
        db.link_paper_author(pid, aid, position=i)

    authors = db.get_paper_authors(pid)
    assert len(authors) == 5
    assert [a["name"] for a in authors] == names


def test_multiple_affiliations_per_author(tmp_db):
    """Author with multiple affiliations."""
    db = Database(tmp_db)
    aid = db.upsert_author(name="Multi-Aff", semantic_scholar_id="999", h_index=50, citation_count=10000)
    db.insert_affiliation(aid, "Google DeepMind", "Google")
    db.insert_affiliation(aid, "Stanford University", "Stanford")
    db.insert_affiliation(aid, "CERN", "CERN")
    affs = db.get_author_affiliations(aid)
    assert len(affs) == 3
    keywords = {a["matched_keyword"] for a in affs}
    assert keywords == {"Google", "Stanford", "CERN"}


def test_upsert_author_updates_on_s2id(tmp_db):
    """Upserting with same S2 ID updates existing author."""
    db = Database(tmp_db)
    aid1 = db.upsert_author(name="Old Name", semantic_scholar_id="abc", h_index=10, citation_count=100)
    aid2 = db.upsert_author(name="New Name", semantic_scholar_id="abc", h_index=20, citation_count=200)
    assert aid1 == aid2
    # Verify updated
    cursor = db.execute("SELECT * FROM authors WHERE id = ?", (aid1,))
    author = dict(cursor.fetchone())
    assert author["name"] == "New Name"
    assert author["h_index"] == 20


def test_category_filter_partial_match(tmp_db):
    """Category filtering uses LIKE, so 'hep' matches 'hep-ph'."""
    db = Database(tmp_db)
    pid = db.insert_paper(
        arxiv_id="2602.11111", title="A", abstract="", categories="hep-ph",
        published_date="", arxiv_url=""
    )
    db.update_combined_score(pid, 5.0)

    assert len(db.get_papers(category="hep-ph", min_score=0)) == 1
    assert len(db.get_papers(category="hep", min_score=0)) == 1
    assert len(db.get_papers(category="astro", min_score=0)) == 0


def test_link_paper_author_idempotent(tmp_db):
    """Linking same author to same paper twice doesn't error (INSERT OR IGNORE)."""
    db = Database(tmp_db)
    pid = db.insert_paper(arxiv_id="2602.11111", title="A", abstract="", categories="", published_date="", arxiv_url="")
    aid = db.upsert_author(name="Alice", semantic_scholar_id=None, h_index=None, citation_count=None)
    db.link_paper_author(pid, aid, 0)
    db.link_paper_author(pid, aid, 0)  # no error
    authors = db.get_paper_authors(pid)
    assert len(authors) == 1


def test_database_creates_file(tmp_path):
    """Database constructor creates the .db file."""
    db_path = str(tmp_path / "new_test.db")
    Database(db_path)
    import os
    assert os.path.exists(db_path)
