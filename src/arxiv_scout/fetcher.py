import re
import time
import requests
import feedparser
from datetime import datetime, timedelta, timezone
from arxiv_scout.db import Database

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_API_PAGE_SIZE = 100
ARXIV_API_DELAY = 3  # seconds between requests (arXiv policy)

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


def backfill_papers(db: Database, categories: list[str], days: int) -> int:
    """Fetch papers from the last N days using the arXiv Search API.

    Unlike fetch_papers() which uses the RSS feed (today only), this queries
    the arXiv Search API with a submittedDate range filter and paginates
    through all results.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    date_from = start.strftime("%Y%m%d")
    date_to = end.strftime("%Y%m%d")

    total_new = 0
    for category in categories:
        query = f"cat:{category} AND submittedDate:[{date_from} TO {date_to}]"
        offset = 0

        while True:
            params = {
                "search_query": query,
                "start": offset,
                "max_results": ARXIV_API_PAGE_SIZE,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)

            total_results = int(feed.feed.get("opensearch_totalresults", 0))
            if not feed.entries:
                break

            for entry in feed.entries:
                # Extract arXiv ID from entry id (e.g. http://arxiv.org/abs/2602.23348v1)
                match = re.search(r'(\d{4}\.\d{4,5})', entry.get("id", ""))
                if not match:
                    continue
                arxiv_id = match.group(1)

                # Categories from tags
                tags = entry.get("tags", [])
                cats = ",".join(t.get("term", "") for t in tags)

                # Authors
                authors = [a.get("name", "") for a in entry.get("authors", [])]

                # Published date
                published = entry.get("published", "")[:10]

                paper_id = db.insert_paper(
                    arxiv_id=arxiv_id,
                    title=" ".join(entry.get("title", "").split()),
                    abstract=entry.get("summary", "").strip(),
                    categories=cats,
                    published_date=published,
                    arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                )
                if paper_id is not None:
                    total_new += 1
                    for i, name in enumerate(authors):
                        author_id = db.upsert_author(
                            name=name, semantic_scholar_id=None,
                            h_index=None, citation_count=None,
                        )
                        db.link_paper_author(paper_id, author_id, position=i)

            offset += ARXIV_API_PAGE_SIZE
            if offset >= total_results:
                break
            time.sleep(ARXIV_API_DELAY)

        print(f"  {category}: {total_results} total, page offset {offset}")
        time.sleep(ARXIV_API_DELAY)

    return total_new
