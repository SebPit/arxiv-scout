import re
import time
import requests
import feedparser
from datetime import datetime, timezone
from arxiv_scout.db import Database

def parse_arxiv_feed(xml_content: str) -> list[dict]:
    """Parse arXiv RSS/RDF feed into paper dicts."""
    feed = feedparser.parse(xml_content)
    papers = []
    for entry in feed.entries:
        link = entry.get("link", "")
        # Extract arXiv ID from link like http://arxiv.org/abs/2602.12345v1
        match = re.search(r'(\d{4}\.\d{4,5})', link)
        if not match:
            continue
        arxiv_id = match.group(1)

        # Clean abstract: strip arXiv metadata prefix
        # Format varies: "arXiv:XXXX.XXXXXvN Announce Type: new \nAbstract: ..."
        #            or: "arXiv:XXXX.XXXXXvN  DD Mon YYYY. ..."
        description = entry.get("description", entry.get("summary", ""))
        abstract = re.sub(r'^arXiv:\S+\s+Announce Type:\s*\w+\s*', '', description).strip()
        abstract = re.sub(r'^Abstract:\s*', '', abstract).strip()
        abstract = re.sub(r'^arXiv:\S+\s+\d{1,2}\s+\w+\s+\d{4}\.\s*', '', abstract).strip()
        # Strip any HTML tags
        abstract = re.sub(r'<[^>]+>', '', abstract).strip()

        # Authors from dc:creator
        authors_str = entry.get("author", entry.get("dc_creator", ""))
        if isinstance(authors_str, str):
            authors = [a.strip() for a in authors_str.split(",") if a.strip()]
        else:
            authors = []

        papers.append({
            "arxiv_id": arxiv_id,
            "title": entry.get("title", "").strip(),
            "abstract": abstract,
            "authors": authors,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "categories": "",  # RSS doesn't reliably give categories
        })
    return papers

def fetch_papers(db: Database, categories: list[str]) -> int:
    """Fetch new papers from arXiv RSS for given categories. Returns count of new papers."""
    total_new = 0
    now = datetime.now(timezone.utc).isoformat()
    for category in categories:
        url = f"https://rss.arxiv.org/rss/{category}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        papers = parse_arxiv_feed(resp.text)
        for paper in papers:
            paper_id = db.insert_paper(
                arxiv_id=paper["arxiv_id"],
                title=paper["title"],
                abstract=paper["abstract"],
                categories=category,
                published_date=now[:10],
                arxiv_url=paper["arxiv_url"],
            )
            if paper_id is not None:
                total_new += 1
                # Store author names (without S2 data yet)
                for i, name in enumerate(paper["authors"]):
                    author_id = db.upsert_author(name=name, semantic_scholar_id=None, h_index=None, citation_count=None)
                    db.link_paper_author(paper_id, author_id, position=i)
        time.sleep(1)  # Be polite to arXiv
    return total_new
