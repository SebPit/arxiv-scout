# ArXiv Scout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a tool that discovers daily arXiv papers (hep-ph, cs.LG), enriches them with Semantic Scholar metadata, scores them using heuristics + Claude Haiku, and serves results via a Flask dashboard.

**Architecture:** ArXiv RSS feeds → SQLite DB → Semantic Scholar enrichment → heuristic + LLM scoring → Flask web dashboard. Daily cron + manual CLI trigger. All config in YAML.

**Tech Stack:** Python 3.13, venv, feedparser, requests, anthropic, flask, pyyaml, apscheduler, sqlite3 (stdlib), pytest

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/arxiv_scout/__init__.py`
- Create: `src/arxiv_scout/__main__.py`
- Create: `config.yaml`
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create venv and project structure**

```bash
cd /home/pitz/Programming/arxiv-scout
python3 -m venv .venv
source .venv/bin/activate
mkdir -p src/arxiv_scout tests
```

**Step 2: Create `requirements.txt`**

```
feedparser>=6.0
requests>=2.31
anthropic>=0.49
flask>=3.0
pyyaml>=6.0
apscheduler>=3.10
pytest>=8.0
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Create `pyproject.toml`**

```toml
[project]
name = "arxiv-scout"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 5: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
*.db
.env
```

**Step 6: Create `config.yaml`**

Full config from design doc (categories, affiliation_keywords, scoring weights, API settings, server settings, schedule).

**Step 7: Create `src/arxiv_scout/__init__.py`** (empty)

**Step 8: Create `src/arxiv_scout/__main__.py`**

```python
"""CLI entry point: python -m arxiv_scout"""
import sys

def main():
    print("arxiv-scout: use 'fetch', 'enrich', 'score', 'serve', or 'run-all'")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

**Step 9: Create `tests/__init__.py`** (empty) and `tests/conftest.py`**

```python
import pytest
import sqlite3
import os

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test.db")
```

**Step 10: Verify setup**

```bash
source .venv/bin/activate
python -m arxiv_scout
pytest --co  # collect tests (should find 0, no error)
```

**Step 11: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with venv, dependencies, and config"
```

---

### Task 2: Database Layer

**Files:**
- Create: `src/arxiv_scout/db.py`
- Create: `tests/test_db.py`

**Step 1: Write failing tests for DB initialization and CRUD**

`tests/test_db.py`:
```python
from arxiv_scout.db import Database

def test_init_creates_tables(tmp_db):
    db = Database(tmp_db)
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
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
    id1 = db.insert_paper(arxiv_id="2602.12345", title="A", abstract="", categories="", published_date="", arxiv_url="")
    id2 = db.insert_paper(arxiv_id="2602.12345", title="B", abstract="", categories="", published_date="", arxiv_url="")
    assert id2 is None  # duplicate skipped

def test_insert_author_and_link(tmp_db):
    db = Database(tmp_db)
    paper_id = db.insert_paper(arxiv_id="2602.12345", title="A", abstract="", categories="", published_date="", arxiv_url="")
    author_id = db.upsert_author(name="Alice Smith", semantic_scholar_id="123", h_index=25, citation_count=5000)
    db.link_paper_author(paper_id, author_id, position=0)
    authors = db.get_paper_authors(paper_id)
    assert len(authors) == 1
    assert authors[0]["name"] == "Alice Smith"

def test_insert_affiliation(tmp_db):
    db = Database(tmp_db)
    author_id = db.upsert_author(name="Bob", semantic_scholar_id=None, h_index=None, citation_count=None)
    db.insert_affiliation(author_id, "Google DeepMind", "Google")
    affiliations = db.get_author_affiliations(author_id)
    assert len(affiliations) == 1
    assert affiliations[0]["matched_keyword"] == "Google"

def test_get_unenriched_papers(tmp_db):
    db = Database(tmp_db)
    db.insert_paper(arxiv_id="2602.11111", title="A", abstract="", categories="", published_date="", arxiv_url="")
    db.insert_paper(arxiv_id="2602.22222", title="B", abstract="", categories="", published_date="", arxiv_url="")
    unenriched = db.get_unenriched_papers()
    assert len(unenriched) == 2

def test_get_unscored_papers(tmp_db):
    db = Database(tmp_db)
    db.insert_paper(arxiv_id="2602.11111", title="A", abstract="Abstract", categories="", published_date="", arxiv_url="")
    unscored = db.get_unscored_papers()
    assert len(unscored) == 1

def test_update_scores(tmp_db):
    db = Database(tmp_db)
    paper_id = db.insert_paper(arxiv_id="2602.11111", title="A", abstract="", categories="", published_date="", arxiv_url="")
    db.update_heuristic_score(paper_id, 7.5)
    db.update_llm_score(paper_id, 8.0, "Very novel paper.")
    db.update_combined_score(paper_id, 7.8)
    paper = db.get_paper_by_arxiv_id("2602.11111")
    assert paper["heuristic_score"] == 7.5
    assert paper["llm_score"] == 8.0
    assert paper["llm_summary"] == "Very novel paper."

def test_get_papers_for_dashboard(tmp_db):
    db = Database(tmp_db)
    db.insert_paper(arxiv_id="2602.11111", title="Good Paper", abstract="Abs", categories="hep-ph", published_date="2026-02-27", arxiv_url="")
    db.update_heuristic_score(1, 5.0)
    db.update_combined_score(1, 5.0)
    papers = db.get_papers(min_score=3.0, category="hep-ph", order_by="combined_score", limit=10)
    assert len(papers) == 1
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v
```
Expected: ImportError — `arxiv_scout.db` does not exist.

**Step 3: Implement `src/arxiv_scout/db.py`**

Implement `Database` class with:
- `__init__(self, db_path)` — creates connection, runs `_init_tables()`
- `_init_tables()` — CREATE TABLE IF NOT EXISTS for all 4 tables
- `execute(sql, params=())` — wrapper around cursor.execute
- `insert_paper(...)` — INSERT OR IGNORE, return lastrowid or None
- `get_paper_by_arxiv_id(arxiv_id)` — SELECT, return dict
- `upsert_author(...)` — INSERT OR REPLACE, return author id
- `link_paper_author(paper_id, author_id, position)`
- `insert_affiliation(author_id, institution_name, matched_keyword)`
- `get_paper_authors(paper_id)` — JOIN query
- `get_author_affiliations(author_id)`
- `get_unenriched_papers()` — papers with no linked authors
- `get_unscored_papers()` — papers with heuristic_score IS NULL
- `update_heuristic_score(paper_id, score)`
- `update_llm_score(paper_id, score, summary)`
- `update_combined_score(paper_id, score)`
- `get_papers(min_score, category, order_by, limit, offset)` — dashboard query
- `mark_enriched(paper_id)` — set a flag so we know enrichment was attempted

Use `row_factory = sqlite3.Row` for dict-like access.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/arxiv_scout/db.py tests/test_db.py
git commit -m "feat: database layer with CRUD operations and tests"
```

---

### Task 3: Configuration Loader

**Files:**
- Create: `src/arxiv_scout/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing tests**

`tests/test_config.py`:
```python
from arxiv_scout.config import load_config

def test_load_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
categories:
  - hep-ph
affiliation_keywords:
  - Google
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
    config = load_config(str(cfg_file))
    assert config["categories"] == ["hep-ph"]
    assert "Google" in config["affiliation_keywords"]
    assert config["scoring"]["heuristic_weight"] == 0.4

def test_load_config_env_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key-123")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
categories: [hep-ph]
affiliation_keywords: []
scoring:
  heuristic_weight: 0.4
  llm_weight: 0.6
  min_heuristic_score_for_llm: 3
  max_papers_per_day: 100
anthropic:
  model: claude-haiku-4-5-20251001
semantic_scholar:
  api_key: ${SEMANTIC_SCHOLAR_API_KEY}
server:
  port: 5000
  host: 127.0.0.1
""")
    config = load_config(str(cfg_file))
    assert config["semantic_scholar"]["api_key"] == "test-key-123"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

**Step 3: Implement `src/arxiv_scout/config.py`**

- `load_config(path)` — read YAML, substitute `${ENV_VAR}` patterns with `os.environ.get()`, return dict.
- Use `re.sub` for env var substitution on the raw YAML string before parsing.

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add src/arxiv_scout/config.py tests/test_config.py
git commit -m "feat: YAML config loader with env var substitution"
```

---

### Task 4: ArXiv RSS Fetcher

**Files:**
- Create: `src/arxiv_scout/fetcher.py`
- Create: `tests/test_fetcher.py`
- Create: `tests/fixtures/sample_rss.xml`

**Step 1: Create a sample RSS fixture**

`tests/fixtures/sample_rss.xml` — a minimal ArXiv RSS feed with 2 entries. Copy realistic structure from actual ArXiv RSS (title, link, description with abstract, dc:creator for authors, arxiv category).

**Step 2: Write failing tests**

`tests/test_fetcher.py`:
```python
from arxiv_scout.fetcher import parse_arxiv_feed, fetch_papers
from unittest.mock import patch, MagicMock
import os

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_rss.xml")

def test_parse_arxiv_feed():
    with open(FIXTURE_PATH) as f:
        xml_content = f.read()
    papers = parse_arxiv_feed(xml_content)
    assert len(papers) >= 1
    paper = papers[0]
    assert "arxiv_id" in paper
    assert "title" in paper
    assert "abstract" in paper
    assert "authors" in paper
    assert isinstance(paper["authors"], list)

def test_parse_extracts_arxiv_id():
    with open(FIXTURE_PATH) as f:
        xml_content = f.read()
    papers = parse_arxiv_feed(xml_content)
    # arXiv IDs should be like "2602.12345"
    for paper in papers:
        assert paper["arxiv_id"]  # not empty

def test_fetch_papers_integration(tmp_db):
    """Test that fetch_papers stores results in DB."""
    from arxiv_scout.db import Database
    db = Database(tmp_db)
    with open(FIXTURE_PATH) as f:
        xml_content = f.read()
    with patch("arxiv_scout.fetcher.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = xml_content
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        count = fetch_papers(db, categories=["hep-ph"])
    assert count >= 1
```

**Step 3: Run tests to verify failure**

**Step 4: Implement `src/arxiv_scout/fetcher.py`**

- `parse_arxiv_feed(xml_content: str) -> list[dict]` — use `feedparser.parse()`, extract arxiv_id from link, title, abstract from description/summary, authors from author field.
- `fetch_papers(db: Database, categories: list[str]) -> int` — for each category, GET `https://rss.arxiv.org/rss/{category}`, parse feed, insert into DB, return count of new papers.
- Handle: link format `http://arxiv.org/abs/2602.12345v1` → extract `2602.12345`.
- Strip HTML from descriptions (arXiv RSS includes `<p>` tags in abstracts).

**Step 5: Run tests, verify pass**

**Step 6: Commit**

```bash
git add src/arxiv_scout/fetcher.py tests/test_fetcher.py tests/fixtures/
git commit -m "feat: ArXiv RSS fetcher with feed parsing and DB storage"
```

---

### Task 5: Semantic Scholar Enricher

**Files:**
- Create: `src/arxiv_scout/enricher.py`
- Create: `tests/test_enricher.py`
- Create: `tests/fixtures/semantic_scholar_batch.json`

**Step 1: Create a Semantic Scholar API response fixture**

`tests/fixtures/semantic_scholar_batch.json` — realistic batch response with 2 papers, each having authors with affiliations and h-index.

**Step 2: Write failing tests**

`tests/test_enricher.py`:
```python
from arxiv_scout.enricher import enrich_papers, match_affiliations, parse_s2_response
from arxiv_scout.db import Database
from unittest.mock import patch, MagicMock
import json, os

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "semantic_scholar_batch.json")

def test_parse_s2_response():
    with open(FIXTURE_PATH) as f:
        data = json.load(f)
    result = parse_s2_response(data[0])
    assert "authors" in result
    assert len(result["authors"]) > 0
    assert "name" in result["authors"][0]

def test_match_affiliations():
    affiliations = ["Google DeepMind", "University of Oxford"]
    keywords = ["Google", "DeepMind", "OpenAI"]
    matches = match_affiliations(affiliations, keywords)
    assert "Google" in matches or "DeepMind" in matches

def test_match_affiliations_no_match():
    affiliations = ["University of Nowhere"]
    keywords = ["Google", "OpenAI"]
    matches = match_affiliations(affiliations, keywords)
    assert len(matches) == 0

def test_enrich_papers_stores_authors(tmp_db):
    db = Database(tmp_db)
    db.insert_paper(arxiv_id="2602.12345", title="A", abstract="", categories="", published_date="", arxiv_url="")
    with open(FIXTURE_PATH) as f:
        fixture_data = json.load(f)
    with patch("arxiv_scout.enricher.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = fixture_data
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        enrich_papers(db, keywords=["Google", "OpenAI"])
    authors = db.get_paper_authors(1)
    assert len(authors) > 0
```

**Step 3: Run tests to verify failure**

**Step 4: Implement `src/arxiv_scout/enricher.py`**

- `parse_s2_response(paper_data: dict) -> dict` — extract authors (name, affiliations, hIndex, citationCount), paper citationCount.
- `match_affiliations(affiliations: list[str], keywords: list[str]) -> list[str]` — case-insensitive substring matching, return matched keywords.
- `enrich_papers(db, keywords, api_key=None, batch_size=500)`:
  1. Get unenriched papers from DB
  2. Build batch of arXiv IDs
  3. POST to `https://api.semanticscholar.org/graph/v1/paper/batch` with `fields=authors.name,authors.affiliations,authors.hIndex,authors.citationCount,citationCount`
  4. Parse response, upsert authors, link to papers, insert affiliations
  5. Mark papers as enriched
- Handle: S2 may return null for some papers (not indexed yet — new papers take 1-3 days).
- Rate limiting: respect 1 req/s.

**Step 5: Run tests, verify pass**

**Step 6: Commit**

```bash
git add src/arxiv_scout/enricher.py tests/test_enricher.py tests/fixtures/semantic_scholar_batch.json
git commit -m "feat: Semantic Scholar enricher with batch API and affiliation matching"
```

---

### Task 6: Heuristic Scorer

**Files:**
- Create: `src/arxiv_scout/scorer.py`
- Create: `tests/test_scorer.py`

**Step 1: Write failing tests**

`tests/test_scorer.py`:
```python
from arxiv_scout.scorer import compute_heuristic_score

def test_heuristic_high_score():
    """Paper with Google affiliation + high h-index author should score high."""
    score = compute_heuristic_score(
        matched_keywords=["Google", "DeepMind"],
        max_h_index=80,
        citation_count=0,  # new paper, no citations yet
    )
    assert score >= 7.0

def test_heuristic_low_score():
    """Paper with no known affiliations and low h-index should score low."""
    score = compute_heuristic_score(
        matched_keywords=[],
        max_h_index=3,
        citation_count=0,
    )
    assert score <= 3.0

def test_heuristic_medium_score():
    """Paper from known institution but moderate h-index."""
    score = compute_heuristic_score(
        matched_keywords=["MIT"],
        max_h_index=20,
        citation_count=0,
    )
    assert 3.0 <= score <= 7.0

def test_heuristic_score_bounds():
    """Score should always be in [0, 10]."""
    score_low = compute_heuristic_score(matched_keywords=[], max_h_index=0, citation_count=0)
    score_high = compute_heuristic_score(matched_keywords=["Google", "OpenAI", "DeepMind"], max_h_index=150, citation_count=1000)
    assert 0 <= score_low <= 10
    assert 0 <= score_high <= 10
```

**Step 2: Run tests to verify failure**

**Step 3: Implement heuristic scoring in `src/arxiv_scout/scorer.py`**

```python
def compute_heuristic_score(matched_keywords: list[str], max_h_index: int, citation_count: int) -> float:
    """Score 0-10 based on institutional affiliation, author impact, and citations."""
    # Affiliation score: 0-4 (each unique keyword match adds points, diminishing)
    affiliation_score = min(4.0, len(matched_keywords) * 2.0)

    # h-index score: 0-4 (sigmoid-like scaling, saturates around h=60)
    h_score = min(4.0, (max_h_index / 60.0) * 4.0) if max_h_index else 0.0

    # Citation score: 0-2 (log scale, mostly 0 for new papers)
    import math
    cite_score = min(2.0, math.log1p(citation_count) / math.log1p(100) * 2.0) if citation_count else 0.0

    return min(10.0, max(0.0, affiliation_score + h_score + cite_score))
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add src/arxiv_scout/scorer.py tests/test_scorer.py
git commit -m "feat: heuristic scoring based on affiliations, h-index, citations"
```

---

### Task 7: LLM Scorer (Claude Haiku)

**Files:**
- Modify: `src/arxiv_scout/scorer.py`
- Create: `tests/test_llm_scorer.py`

**Step 1: Write failing tests**

`tests/test_llm_scorer.py`:
```python
from arxiv_scout.scorer import score_with_llm, parse_llm_response
from unittest.mock import patch, MagicMock

def test_parse_llm_response_valid():
    response_text = '{"score": 8, "summary": "Novel approach to unfolding using normalizing flows."}'
    score, summary = parse_llm_response(response_text)
    assert score == 8
    assert "unfolding" in summary.lower()

def test_parse_llm_response_invalid_json():
    response_text = "This is not JSON"
    score, summary = parse_llm_response(response_text)
    assert score is None
    assert summary is None

def test_parse_llm_response_out_of_range():
    response_text = '{"score": 15, "summary": "Amazing."}'
    score, summary = parse_llm_response(response_text)
    assert score == 10  # clamped

def test_score_with_llm():
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text='{"score": 7, "summary": "Interesting new method."}')]
    mock_client.messages.create.return_value = mock_message

    score, summary = score_with_llm(
        client=mock_client,
        model="claude-haiku-4-5-20251001",
        title="A Novel Method",
        abstract="We propose a new approach to X.",
        categories="cs.LG",
    )
    assert score == 7
    assert "method" in summary.lower()
    mock_client.messages.create.assert_called_once()
```

**Step 2: Run tests to verify failure**

**Step 3: Add LLM scoring functions to `src/arxiv_scout/scorer.py`**

- `parse_llm_response(text: str) -> tuple[int|None, str|None]` — parse JSON `{"score": N, "summary": "..."}`, clamp score to 0-10.
- `score_with_llm(client, model, title, abstract, categories) -> tuple[int|None, str|None]` — call Claude with a system prompt:

```
You are a research paper evaluator. Rate this paper's novelty and potential impact.
Respond with JSON only: {"score": <1-10>, "summary": "<2-3 sentence assessment>"}
Score guide: 1-3=incremental, 4-6=solid contribution, 7-8=significant, 9-10=potentially groundbreaking
```

- `score_papers(db, config)` — orchestrate: get unscored papers, compute heuristic scores, LLM-score those above threshold, compute combined scores.

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add src/arxiv_scout/scorer.py tests/test_llm_scorer.py
git commit -m "feat: LLM scorer using Claude Haiku for novelty assessment"
```

---

### Task 8: Pipeline Orchestrator & CLI

**Files:**
- Create: `src/arxiv_scout/pipeline.py`
- Modify: `src/arxiv_scout/__main__.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write failing tests**

`tests/test_pipeline.py`:
```python
from arxiv_scout.pipeline import run_pipeline
from unittest.mock import patch, MagicMock

def test_run_pipeline_calls_stages(tmp_db, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
categories: [hep-ph]
affiliation_keywords: [Google]
scoring:
  heuristic_weight: 0.4
  llm_weight: 0.6
  min_heuristic_score_for_llm: 3
  max_papers_per_day: 10
anthropic:
  model: claude-haiku-4-5-20251001
server:
  port: 5000
  host: 127.0.0.1
""")
    with patch("arxiv_scout.pipeline.fetch_papers") as mock_fetch, \
         patch("arxiv_scout.pipeline.enrich_papers") as mock_enrich, \
         patch("arxiv_scout.pipeline.score_papers") as mock_score:
        mock_fetch.return_value = 5
        run_pipeline(str(config_file), db_path=tmp_db)
        mock_fetch.assert_called_once()
        mock_enrich.assert_called_once()
        mock_score.assert_called_once()
```

**Step 2: Run tests to verify failure**

**Step 3: Implement `src/arxiv_scout/pipeline.py`**

```python
def run_pipeline(config_path: str, db_path: str = "arxiv_scout.db"):
    """Run the full fetch → enrich → score pipeline."""
    config = load_config(config_path)
    db = Database(db_path)
    count = fetch_papers(db, config["categories"])
    print(f"Fetched {count} new papers")
    enrich_papers(db, config["affiliation_keywords"],
                  api_key=config.get("semantic_scholar", {}).get("api_key"))
    score_papers(db, config)
    print("Pipeline complete")
```

**Step 4: Update `src/arxiv_scout/__main__.py`**

Add argparse CLI with subcommands: `fetch`, `enrich`, `score`, `serve`, `run-all`.
Each subcommand accepts `--config` (default: `config.yaml`) and `--db` (default: `arxiv_scout.db`).

**Step 5: Run tests, verify pass**

**Step 6: Commit**

```bash
git add src/arxiv_scout/pipeline.py src/arxiv_scout/__main__.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrator and CLI with subcommands"
```

---

### Task 9: Flask Web Dashboard — Backend

**Files:**
- Create: `src/arxiv_scout/app.py`
- Create: `tests/test_app.py`

**Step 1: Write failing tests**

`tests/test_app.py`:
```python
import pytest
from arxiv_scout.app import create_app
from arxiv_scout.db import Database

@pytest.fixture
def app(tmp_db):
    db = Database(tmp_db)
    # Seed test data
    pid = db.insert_paper(
        arxiv_id="2602.12345", title="Test Paper", abstract="An interesting abstract.",
        categories="hep-ph", published_date="2026-02-27",
        arxiv_url="https://arxiv.org/abs/2602.12345"
    )
    db.update_heuristic_score(pid, 7.0)
    db.update_llm_score(pid, 8.0, "Novel approach.")
    db.update_combined_score(pid, 7.6)
    aid = db.upsert_author(name="Alice", semantic_scholar_id="1", h_index=50, citation_count=10000)
    db.link_paper_author(pid, aid, 0)
    db.insert_affiliation(aid, "Google DeepMind", "Google")

    app = create_app(db_path=tmp_db)
    app.config["TESTING"] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def test_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200

def test_index_contains_paper(client):
    response = client.get("/")
    assert b"Test Paper" in response.data

def test_filter_by_category(client):
    response = client.get("/?category=hep-ph")
    assert b"Test Paper" in response.data
    response = client.get("/?category=cs.LG")
    assert b"Test Paper" not in response.data

def test_filter_by_min_score(client):
    response = client.get("/?min_score=9")
    assert b"Test Paper" not in response.data

def test_api_papers_json(client):
    response = client.get("/api/papers")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["papers"]) == 1
```

**Step 2: Run tests to verify failure**

**Step 3: Implement `src/arxiv_scout/app.py`**

```python
from flask import Flask, render_template, request, jsonify
from arxiv_scout.db import Database

def create_app(db_path="arxiv_scout.db"):
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    db = Database(db_path)

    @app.route("/")
    def index():
        category = request.args.get("category")
        min_score = request.args.get("min_score", type=float, default=0)
        sort = request.args.get("sort", "combined_score")
        page = request.args.get("page", type=int, default=1)
        per_page = 50
        papers = db.get_papers(
            min_score=min_score, category=category,
            order_by=sort, limit=per_page, offset=(page-1)*per_page
        )
        return render_template("index.html", papers=papers, filters={...})

    @app.route("/api/papers")
    def api_papers():
        papers = db.get_papers(min_score=0, limit=100)
        return jsonify({"papers": [dict(p) for p in papers]})

    return app
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add src/arxiv_scout/app.py tests/test_app.py
git commit -m "feat: Flask app with paper listing, filtering, and JSON API"
```

---

### Task 10: Flask Web Dashboard — Frontend Templates

**Files:**
- Create: `src/templates/base.html`
- Create: `src/templates/index.html`
- Create: `src/static/style.css`

**Step 1: Create `src/templates/base.html`**

Minimal HTML5 layout with:
- Clean, modern CSS (no framework needed — keep it light)
- Header: "ArXiv Scout" + nav
- Content block
- Dark/light color scheme

**Step 2: Create `src/templates/index.html`**

Extends base. Contains:
- Filter bar: category dropdown, min score slider, sort dropdown, date range
- Paper cards: title (linked to arXiv), score badge, author list with affiliation highlights, category tags, published date
- Expandable abstract + LLM summary on click (use `<details>` element — no JS needed)
- Pagination

**Step 3: Create `src/static/style.css`**

Clean, readable styles. Score badges color-coded (green ≥7, yellow 4-7, red <4). Affiliation keyword matches highlighted.

**Step 4: Manual verification**

```bash
cd /home/pitz/Programming/arxiv-scout
source .venv/bin/activate
python -c "
from arxiv_scout.db import Database
db = Database('test_visual.db')
pid = db.insert_paper(arxiv_id='2602.99999', title='Attention Is All You Need (Again)', abstract='We present Transformer v2...', categories='cs.LG', published_date='2026-02-27', arxiv_url='https://arxiv.org/abs/2602.99999')
db.update_heuristic_score(pid, 9.0)
db.update_llm_score(pid, 9.5, 'Groundbreaking architecture improvement.')
db.update_combined_score(pid, 9.3)
aid = db.upsert_author(name='Ashish Vaswani', semantic_scholar_id='1', h_index=45, citation_count=80000)
db.link_paper_author(pid, aid, 0)
db.insert_affiliation(aid, 'Google Brain', 'Google')
"
python -m arxiv_scout serve --db test_visual.db
# Open http://127.0.0.1:5000 in browser, verify layout
rm test_visual.db
```

**Step 5: Commit**

```bash
git add src/templates/ src/static/
git commit -m "feat: dashboard HTML templates with filters, score badges, and styling"
```

---

### Task 11: Scheduler Integration

**Files:**
- Create: `src/arxiv_scout/scheduler.py`
- Modify: `src/arxiv_scout/__main__.py`

**Step 1: Implement `src/arxiv_scout/scheduler.py`**

```python
from apscheduler.schedulers.background import BackgroundScheduler
from arxiv_scout.pipeline import run_pipeline

def start_scheduler(config_path: str, db_path: str, fetch_time: str = "06:00"):
    hour, minute = map(int, fetch_time.split(":"))
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_pipeline, "cron", hour=hour, minute=minute,
        args=[config_path, db_path], id="daily_fetch"
    )
    scheduler.start()
    return scheduler
```

**Step 2: Update `__main__.py`** — integrate scheduler into `serve` subcommand

When `serve` runs, start scheduler in background + Flask app in foreground.

**Step 3: Test manually**

```bash
python -m arxiv_scout serve --config config.yaml
# Verify scheduler logs "next run at ..."
```

**Step 4: Commit**

```bash
git add src/arxiv_scout/scheduler.py src/arxiv_scout/__main__.py
git commit -m "feat: APScheduler integration for daily automated fetching"
```

---

### Task 12: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
"""End-to-end test using mocked external APIs."""
from unittest.mock import patch, MagicMock
from arxiv_scout.pipeline import run_pipeline
from arxiv_scout.db import Database
import os

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

def test_full_pipeline(tmp_db, tmp_path):
    # Setup config
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""...""")  # full config

    with open(os.path.join(FIXTURES, "sample_rss.xml")) as f:
        rss_data = f.read()
    with open(os.path.join(FIXTURES, "semantic_scholar_batch.json")) as f:
        s2_data = json.load(f)

    mock_rss = MagicMock(text=rss_data, status_code=200)
    mock_s2 = MagicMock()
    mock_s2.json.return_value = s2_data
    mock_s2.status_code = 200

    mock_llm_msg = MagicMock()
    mock_llm_msg.content = [MagicMock(text='{"score": 7, "summary": "Good paper."}')]
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = mock_llm_msg

    with patch("arxiv_scout.fetcher.requests.get", return_value=mock_rss), \
         patch("arxiv_scout.enricher.requests.post", return_value=mock_s2), \
         patch("arxiv_scout.scorer.anthropic.Anthropic", return_value=mock_anthropic):
        run_pipeline(str(config_file), db_path=tmp_db)

    db = Database(tmp_db)
    papers = db.get_papers(min_score=0, limit=100)
    assert len(papers) >= 1
    assert papers[0]["combined_score"] is not None
```

**Step 2: Run all tests**

```bash
pytest tests/ -v
```

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test with mocked APIs"
```

---

### Task 13: Polish & Documentation

**Files:**
- Modify: `config.yaml` — finalize default affiliation keywords
- Create: `src/arxiv_scout/utils.py` — any shared helpers extracted during implementation

**Step 1: Final test run**

```bash
pytest tests/ -v --tb=short
```

**Step 2: Manual end-to-end test with real APIs**

```bash
# Set up API keys
export ANTHROPIC_API_KEY=<your-key>
export SEMANTIC_SCHOLAR_API_KEY=<your-key>  # optional but recommended

# Run pipeline
python -m arxiv_scout run-all --config config.yaml

# Launch dashboard
python -m arxiv_scout serve --config config.yaml
# Open http://127.0.0.1:5000
```

**Step 3: Commit final state**

```bash
git add -A
git commit -m "chore: polish config defaults and final integration verification"
```
