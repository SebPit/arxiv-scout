"""SQLite database layer for ArXiv Scout."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone


class Database:
    """Thin wrapper around SQLite for paper/author/affiliation storage."""

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_tables()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
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
                scored_at TEXT,
                enrichment_attempted_at TEXT,
                enrichment_found INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS authors (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                semantic_scholar_id TEXT UNIQUE,
                h_index INTEGER,
                citation_count INTEGER
            );

            CREATE TABLE IF NOT EXISTS paper_authors (
                paper_id INTEGER REFERENCES papers(id),
                author_id INTEGER REFERENCES authors(id),
                position INTEGER,
                PRIMARY KEY (paper_id, author_id)
            );

            CREATE TABLE IF NOT EXISTS affiliations (
                id INTEGER PRIMARY KEY,
                author_id INTEGER REFERENCES authors(id),
                institution_name TEXT,
                matched_keyword TEXT
            );
            """
        )

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL, auto-commit, and return the cursor."""
        cursor = self.conn.execute(sql, params)
        self.conn.commit()
        return cursor

    # ------------------------------------------------------------------
    # Papers
    # ------------------------------------------------------------------

    def insert_paper(
        self,
        arxiv_id: str,
        title: str,
        abstract: str,
        categories: str,
        published_date: str,
        arxiv_url: str,
    ) -> int | None:
        """INSERT OR IGNORE a paper. Returns lastrowid or None if duplicate."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.execute(
            """
            INSERT OR IGNORE INTO papers
                (arxiv_id, title, abstract, categories, published_date, arxiv_url, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (arxiv_id, title, abstract, categories, published_date, arxiv_url, now),
        )
        if cursor.rowcount == 0:
            return None
        return cursor.lastrowid

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> dict | None:
        """Return a paper as a dict, or None if not found."""
        cursor = self.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Authors
    # ------------------------------------------------------------------

    def upsert_author(
        self,
        name: str,
        semantic_scholar_id: str | None,
        h_index: int | None,
        citation_count: int | None,
    ) -> int:
        """Upsert an author. If semantic_scholar_id is None, just INSERT.

        Returns the author id.
        """
        if semantic_scholar_id is None:
            cursor = self.execute(
                """
                INSERT INTO authors (name, semantic_scholar_id, h_index, citation_count)
                VALUES (?, NULL, ?, ?)
                """,
                (name, h_index, citation_count),
            )
            return cursor.lastrowid
        else:
            # Try to find existing author by semantic_scholar_id
            existing = self.execute(
                "SELECT id FROM authors WHERE semantic_scholar_id = ?",
                (semantic_scholar_id,),
            ).fetchone()
            if existing:
                self.execute(
                    """
                    UPDATE authors
                    SET name = ?, h_index = ?, citation_count = ?
                    WHERE semantic_scholar_id = ?
                    """,
                    (name, h_index, citation_count, semantic_scholar_id),
                )
                return existing["id"]
            else:
                cursor = self.execute(
                    """
                    INSERT INTO authors (name, semantic_scholar_id, h_index, citation_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, semantic_scholar_id, h_index, citation_count),
                )
                return cursor.lastrowid

    # ------------------------------------------------------------------
    # Paper-Author linking
    # ------------------------------------------------------------------

    def link_paper_author(
        self, paper_id: int, author_id: int, position: int
    ) -> None:
        """Link a paper to an author at the given position."""
        self.execute(
            """
            INSERT OR IGNORE INTO paper_authors (paper_id, author_id, position)
            VALUES (?, ?, ?)
            """,
            (paper_id, author_id, position),
        )

    def get_paper_authors(self, paper_id: int) -> list[dict]:
        """Return authors for a paper, joined with the authors table."""
        cursor = self.execute(
            """
            SELECT a.*, pa.position
            FROM authors a
            JOIN paper_authors pa ON a.id = pa.author_id
            WHERE pa.paper_id = ?
            ORDER BY pa.position
            """,
            (paper_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Affiliations
    # ------------------------------------------------------------------

    def insert_affiliation(
        self, author_id: int, institution_name: str, matched_keyword: str
    ) -> int:
        """Insert an affiliation record. Returns the new row id."""
        cursor = self.execute(
            """
            INSERT INTO affiliations (author_id, institution_name, matched_keyword)
            VALUES (?, ?, ?)
            """,
            (author_id, institution_name, matched_keyword),
        )
        return cursor.lastrowid

    def get_author_affiliations(self, author_id: int) -> list[dict]:
        """Return all affiliations for an author."""
        cursor = self.execute(
            "SELECT * FROM affiliations WHERE author_id = ?", (author_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Enrichment queries
    # ------------------------------------------------------------------

    def get_unenriched_papers(self) -> list[dict]:
        """Return papers eligible for Semantic Scholar enrichment.

        A paper is eligible if:
        - enrichment_attempted_at IS NULL (never tried), OR
        - enrichment_found = 0 AND last attempt was >24h ago
          AND fetched_at is within the last 7 days (don't retry forever).
        """
        now = datetime.now(timezone.utc)
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        cutoff_7d = (now - timedelta(days=7)).isoformat()
        cursor = self.execute(
            """
            SELECT * FROM papers
            WHERE enrichment_attempted_at IS NULL
               OR (
                   enrichment_found = 0
                   AND enrichment_attempted_at < ?
                   AND fetched_at > ?
               )
            """,
            (cutoff_24h, cutoff_7d),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Scoring queries
    # ------------------------------------------------------------------

    def get_unscored_papers(self) -> list[dict]:
        """Return papers that have not yet received a heuristic score."""
        cursor = self.execute(
            "SELECT * FROM papers WHERE heuristic_score IS NULL"
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_heuristic_score(self, paper_id: int, score: float) -> None:
        """Set the heuristic score for a paper."""
        self.execute(
            "UPDATE papers SET heuristic_score = ? WHERE id = ?",
            (score, paper_id),
        )

    def update_llm_score(
        self, paper_id: int, score: float, summary: str
    ) -> None:
        """Set the LLM score and summary, and record scored_at timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        self.execute(
            """
            UPDATE papers SET llm_score = ?, llm_summary = ?, scored_at = ?
            WHERE id = ?
            """,
            (score, summary, now, paper_id),
        )

    def update_combined_score(self, paper_id: int, score: float) -> None:
        """Set the combined (final) score for a paper."""
        self.execute(
            "UPDATE papers SET combined_score = ? WHERE id = ?",
            (score, paper_id),
        )

    # ------------------------------------------------------------------
    # Enrichment bookkeeping
    # ------------------------------------------------------------------

    def mark_enrichment_attempted(
        self, paper_id: int, found: bool
    ) -> None:
        """Record that enrichment was attempted for a paper."""
        now = datetime.now(timezone.utc).isoformat()
        self.execute(
            """
            UPDATE papers
            SET enrichment_attempted_at = ?, enrichment_found = ?
            WHERE id = ?
            """,
            (now, int(found), paper_id),
        )

    # ------------------------------------------------------------------
    # Dashboard query
    # ------------------------------------------------------------------

    def get_papers(
        self,
        min_score: float = 0,
        category: str | None = None,
        order_by: str = "combined_score",
        limit: int = 50,
        offset: int = 0,
        since: str | None = None,
    ) -> list[dict]:
        """Query papers for the dashboard.

        Returns papers ordered descending by the chosen column, filtered by
        minimum combined_score and optionally by category substring match.
        If *since* is given (ISO date string like '2026-03-01'), only papers
        fetched on or after that date are included.
        """
        # Whitelist allowed order_by columns to prevent SQL injection
        allowed_order = {
            "combined_score",
            "heuristic_score",
            "llm_score",
            "published_date",
            "fetched_at",
            "citation_count",
        }
        if order_by not in allowed_order:
            order_by = "combined_score"

        conditions = ["combined_score >= ?"]
        params: list = [min_score]

        if category is not None:
            conditions.append("categories LIKE ?")
            params.append(f"%{category}%")

        if since is not None:
            conditions.append("fetched_at >= ?")
            params.append(since)

        where_clause = " AND ".join(conditions)
        params.extend([limit, offset])

        cursor = self.execute(
            f"""
            SELECT * FROM papers
            WHERE {where_clause}
            ORDER BY {order_by} DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params),
        )
        return [dict(row) for row in cursor.fetchall()]
