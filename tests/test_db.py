from arxiv_scout.db import Database


def test_init_creates_tables(tmp_db):
    db = Database(tmp_db)
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {row[0] for row in tables}
    assert "papers" in table_names
    assert "authors" in table_names
    assert "paper_authors" in table_names
    assert "affiliations" in table_names


def test_insert_paper(tmp_db):
    db = Database(tmp_db)
    paper_id = db.insert_paper(
        arxiv_id="2602.12345",
        title="Test Paper",
        abstract="An abstract.",
        categories="hep-ph",
        published_date="2026-02-27",
        arxiv_url="https://arxiv.org/abs/2602.12345",
    )
    assert paper_id is not None
    paper = db.get_paper_by_arxiv_id("2602.12345")
    assert paper["title"] == "Test Paper"


def test_insert_duplicate_paper_skips(tmp_db):
    db = Database(tmp_db)
    id1 = db.insert_paper(
        arxiv_id="2602.12345",
        title="A",
        abstract="",
        categories="",
        published_date="",
        arxiv_url="",
    )
    id2 = db.insert_paper(
        arxiv_id="2602.12345",
        title="B",
        abstract="",
        categories="",
        published_date="",
        arxiv_url="",
    )
    assert id2 is None  # duplicate skipped


def test_insert_author_and_link(tmp_db):
    db = Database(tmp_db)
    paper_id = db.insert_paper(
        arxiv_id="2602.12345",
        title="A",
        abstract="",
        categories="",
        published_date="",
        arxiv_url="",
    )
    author_id = db.upsert_author(
        name="Alice Smith",
        semantic_scholar_id="123",
        h_index=25,
        citation_count=5000,
    )
    db.link_paper_author(paper_id, author_id, position=0)
    authors = db.get_paper_authors(paper_id)
    assert len(authors) == 1
    assert authors[0]["name"] == "Alice Smith"


def test_insert_affiliation(tmp_db):
    db = Database(tmp_db)
    author_id = db.upsert_author(
        name="Bob",
        semantic_scholar_id=None,
        h_index=None,
        citation_count=None,
    )
    db.insert_affiliation(author_id, "Google DeepMind", "Google")
    affiliations = db.get_author_affiliations(author_id)
    assert len(affiliations) == 1
    assert affiliations[0]["matched_keyword"] == "Google"


def test_get_unenriched_papers(tmp_db):
    db = Database(tmp_db)
    db.insert_paper(
        arxiv_id="2602.11111",
        title="A",
        abstract="",
        categories="",
        published_date="",
        arxiv_url="",
    )
    db.insert_paper(
        arxiv_id="2602.22222",
        title="B",
        abstract="",
        categories="",
        published_date="",
        arxiv_url="",
    )
    unenriched = db.get_unenriched_papers()
    assert len(unenriched) == 2


def test_get_unscored_papers(tmp_db):
    db = Database(tmp_db)
    db.insert_paper(
        arxiv_id="2602.11111",
        title="A",
        abstract="Abstract",
        categories="",
        published_date="",
        arxiv_url="",
    )
    unscored = db.get_unscored_papers()
    assert len(unscored) == 1


def test_update_scores(tmp_db):
    db = Database(tmp_db)
    paper_id = db.insert_paper(
        arxiv_id="2602.11111",
        title="A",
        abstract="",
        categories="",
        published_date="",
        arxiv_url="",
    )
    db.update_heuristic_score(paper_id, 7.5)
    db.update_llm_score(paper_id, 8.0, "Very novel paper.")
    db.update_combined_score(paper_id, 7.8)
    paper = db.get_paper_by_arxiv_id("2602.11111")
    assert paper["heuristic_score"] == 7.5
    assert paper["llm_score"] == 8.0
    assert paper["llm_summary"] == "Very novel paper."
    assert paper["combined_score"] == 7.8


def test_mark_enrichment_attempted(tmp_db):
    db = Database(tmp_db)
    paper_id = db.insert_paper(
        arxiv_id="2602.11111",
        title="A",
        abstract="",
        categories="",
        published_date="",
        arxiv_url="",
    )
    db.mark_enrichment_attempted(paper_id, found=False)
    paper = db.get_paper_by_arxiv_id("2602.11111")
    assert paper["enrichment_attempted_at"] is not None
    assert paper["enrichment_found"] == 0


def test_get_papers_for_dashboard(tmp_db):
    db = Database(tmp_db)
    db.insert_paper(
        arxiv_id="2602.11111",
        title="Good Paper",
        abstract="Abs",
        categories="hep-ph",
        published_date="2026-02-27",
        arxiv_url="",
    )
    db.update_heuristic_score(1, 5.0)
    db.update_combined_score(1, 5.0)
    papers = db.get_papers(
        min_score=3.0, category="hep-ph", order_by="combined_score", limit=10
    )
    assert len(papers) == 1


def test_upsert_author_updates_existing(tmp_db):
    """Upserting an author with the same semantic_scholar_id should update."""
    db = Database(tmp_db)
    id1 = db.upsert_author(
        name="Alice",
        semantic_scholar_id="S2-123",
        h_index=10,
        citation_count=100,
    )
    id2 = db.upsert_author(
        name="Alice Smith",
        semantic_scholar_id="S2-123",
        h_index=15,
        citation_count=200,
    )
    # Should be same author row (updated in place)
    authors = db.execute("SELECT * FROM authors").fetchall()
    assert len(authors) == 1
    assert authors[0]["h_index"] == 15
    assert authors[0]["name"] == "Alice Smith"


def test_upsert_author_no_s2id_inserts_each_time(tmp_db):
    """Authors without semantic_scholar_id just get inserted (no dedup key)."""
    db = Database(tmp_db)
    id1 = db.upsert_author(name="Unknown A", semantic_scholar_id=None, h_index=None, citation_count=None)
    id2 = db.upsert_author(name="Unknown B", semantic_scholar_id=None, h_index=None, citation_count=None)
    assert id1 != id2


def test_get_papers_ordering_and_limit(tmp_db):
    """Dashboard query should return papers ordered by score descending with limit."""
    db = Database(tmp_db)
    for i in range(5):
        pid = db.insert_paper(
            arxiv_id=f"2602.{10000 + i}",
            title=f"Paper {i}",
            abstract="",
            categories="hep-ph",
            published_date="2026-02-27",
            arxiv_url="",
        )
        db.update_combined_score(pid, float(i))

    papers = db.get_papers(min_score=0, limit=3)
    assert len(papers) == 3
    # Should be descending
    assert papers[0]["combined_score"] >= papers[1]["combined_score"]
    assert papers[1]["combined_score"] >= papers[2]["combined_score"]


def test_get_papers_with_offset(tmp_db):
    """Dashboard query should support pagination via offset."""
    db = Database(tmp_db)
    for i in range(5):
        pid = db.insert_paper(
            arxiv_id=f"2602.{10000 + i}",
            title=f"Paper {i}",
            abstract="",
            categories="hep-ph",
            published_date="2026-02-27",
            arxiv_url="",
        )
        db.update_combined_score(pid, float(i))

    page1 = db.get_papers(min_score=0, limit=2, offset=0)
    page2 = db.get_papers(min_score=0, limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    # No overlap
    ids1 = {p["arxiv_id"] for p in page1}
    ids2 = {p["arxiv_id"] for p in page2}
    assert ids1.isdisjoint(ids2)


def test_mark_enrichment_found_true(tmp_db):
    """Marking enrichment as found=True should set enrichment_found=1."""
    db = Database(tmp_db)
    paper_id = db.insert_paper(
        arxiv_id="2602.99999",
        title="A",
        abstract="",
        categories="",
        published_date="",
        arxiv_url="",
    )
    db.mark_enrichment_attempted(paper_id, found=True)
    paper = db.get_paper_by_arxiv_id("2602.99999")
    assert paper["enrichment_found"] == 1
    assert paper["enrichment_attempted_at"] is not None
