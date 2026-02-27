# ArXiv Scout — Design Document

**Date:** 2026-02-27
**Goal:** Automatically discover and rank influential/groundbreaking papers from arXiv (hep-ph, cs.LG) using author affiliations, impact metrics, and LLM-based novelty scoring. Serve results via a local web dashboard.

## Architecture

```
ArXiv RSS Feeds ──▶ Fetcher ──▶ SQLite DB ◀── Flask Dashboard
                                    │
                    Semantic Scholar API
                    (affiliations, h-index)
                                    │
                    Claude Haiku (novelty scoring)
```

## Pipeline

1. **Fetch** — Pull daily papers from ArXiv RSS (`hep-ph`, `cs.LG`)
2. **Enrich** — Query Semantic Scholar for author affiliations, h-index, citation counts
3. **Score** — Heuristic pre-filter + Claude Haiku novelty scoring for top candidates
4. **Serve** — Flask + Jinja2 dashboard with search/filter/sort

## Components

### Paper Fetcher (`fetcher.py`)
- Parse ArXiv RSS/Atom feeds for configured categories
- Deduplicate against existing DB entries by arXiv ID
- Store: title, abstract, authors, categories, date, arXiv URL

### Enricher (`enricher.py`)
- Semantic Scholar batch API: `POST /paper/batch` with arXiv IDs
- Fields: `authors.name, authors.affiliations, authors.hIndex, authors.citationCount, citationCount`
- Match affiliations against configurable keyword list
- Rate limit: 1 req/s with API key

### Scorer (`scorer.py`)
- **Heuristic score (0-10):** weighted sum of affiliation matches, max author h-index, citation count
- **LLM score (0-10):** Claude Haiku reads abstract, rates novelty + potential impact
- **Combined score:** `0.4 * heuristic + 0.6 * llm`
- Only LLM-score papers with heuristic score >= 3 (cost control)

### Web Dashboard (`app.py`)
- Flask serving Jinja2 templates
- Paper list: title, authors, score, affiliations, date, category
- Filters: category, date range, min score, affiliation keyword
- Sort: score, date, citations
- Expandable: full abstract + LLM summary
- Links to arXiv abstract/PDF

### Scheduler
- APScheduler for daily runs (configurable time)
- CLI command for manual trigger: `python -m arxiv_scout fetch`

## Data Model (SQLite)

```sql
CREATE TABLE papers (
    id INTEGER PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    categories TEXT,
    published_date TEXT,
    arxiv_url TEXT,
    citation_count INTEGER DEFAULT 0,
    heuristic_score REAL,
    llm_score REAL,
    llm_summary TEXT,
    combined_score REAL,
    fetched_at TEXT,
    scored_at TEXT
);

CREATE TABLE authors (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    semantic_scholar_id TEXT,
    h_index INTEGER,
    citation_count INTEGER
);

CREATE TABLE paper_authors (
    paper_id INTEGER REFERENCES papers(id),
    author_id INTEGER REFERENCES authors(id),
    position INTEGER,
    PRIMARY KEY (paper_id, author_id)
);

CREATE TABLE affiliations (
    id INTEGER PRIMARY KEY,
    author_id INTEGER REFERENCES authors(id),
    institution_name TEXT,
    matched_keyword TEXT
);
```

## Configuration (`config.yaml`)

```yaml
categories:
  - hep-ph
  - cs.LG

affiliation_keywords:
  - Google
  - DeepMind
  - OpenAI
  - Anthropic
  - Meta AI
  - FAIR
  - Microsoft Research
  - CERN
  - MIT
  - Stanford
  - Berkeley
  - Princeton
  - Harvard
  - Caltech
  - ETH Zurich

scoring:
  heuristic_weight: 0.4
  llm_weight: 0.6
  min_heuristic_score_for_llm: 3
  max_papers_per_day: 100

anthropic:
  model: claude-haiku-4-5-20251001

semantic_scholar:
  api_key: ${SEMANTIC_SCHOLAR_API_KEY}
  batch_size: 500

server:
  port: 5000
  host: 127.0.0.1

schedule:
  fetch_time: "06:00"
```

## Cost Estimate
- ~200-500 papers/day across both categories
- Semantic Scholar: free (1 req/s, batch endpoint)
- Claude Haiku: ~100 abstracts/day x ~500 tokens = ~$0.01-0.02/day

## Tech Stack
- Python 3.13 + venv
- `requests`, `feedparser` — RSS/API calls
- `anthropic` — Claude API
- `flask` — web dashboard
- `pyyaml` — configuration
- `apscheduler` — scheduling
- SQLite3 (stdlib) — storage
