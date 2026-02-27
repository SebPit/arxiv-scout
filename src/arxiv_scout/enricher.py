import time
import requests
from datetime import datetime, timezone
from arxiv_scout.db import Database

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "authors.name,authors.affiliations,authors.hIndex,authors.citationCount,citationCount,externalIds"

def match_affiliations(affiliations: list[str], keywords: list[str]) -> list[str]:
    """Match affiliations against keywords (case-insensitive substring)."""
    matched = []
    for kw in keywords:
        kw_lower = kw.lower()
        for aff in affiliations:
            if kw_lower in aff.lower():
                matched.append(kw)
                break  # Don't duplicate the same keyword
    return matched

def parse_s2_response(paper_data: dict) -> dict:
    """Parse a single Semantic Scholar paper response."""
    authors = []
    for a in paper_data.get("authors", []):
        authors.append({
            "name": a.get("name", ""),
            "author_id": a.get("authorId"),
            "affiliations": a.get("affiliations") or [],
            "h_index": a.get("hIndex"),
            "citation_count": a.get("citationCount"),
        })
    return {
        "citation_count": paper_data.get("citationCount", 0),
        "authors": authors,
    }

def enrich_papers(db: Database, keywords: list[str], api_key: str | None = None, batch_size: int = 500):
    """Enrich papers with Semantic Scholar data."""
    papers = db.get_unenriched_papers()
    if not papers:
        return

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    # Process in batches
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        arxiv_ids = [f"ArXiv:{p['arxiv_id']}" for p in batch]

        resp = requests.post(
            S2_BATCH_URL,
            json={"ids": arxiv_ids},
            params={"fields": S2_FIELDS},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        results = resp.json()

        for paper, s2_data in zip(batch, results):
            if s2_data is None:
                # Not indexed yet -- mark attempted but not found
                db.mark_enrichment_attempted(paper["id"], found=False)
                continue

            parsed = parse_s2_response(s2_data)

            # Update paper citation count
            db.execute(
                "UPDATE papers SET citation_count = ? WHERE id = ?",
                (parsed["citation_count"], paper["id"])
            )

            for j, author in enumerate(parsed["authors"]):
                author_id = db.upsert_author(
                    name=author["name"],
                    semantic_scholar_id=author["author_id"],
                    h_index=author["h_index"],
                    citation_count=author["citation_count"],
                )
                db.link_paper_author(paper["id"], author_id, position=j)

                # Match and store affiliations
                matched = match_affiliations(author["affiliations"], keywords)
                for aff in author["affiliations"]:
                    kw = next((k for k in matched if k.lower() in aff.lower()), None)
                    db.insert_affiliation(author_id, aff, kw)

            db.mark_enrichment_attempted(paper["id"], found=True)

        time.sleep(1)  # Rate limiting
