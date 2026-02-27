from arxiv_scout.config import load_config
from arxiv_scout.db import Database
from arxiv_scout.fetcher import fetch_papers
from arxiv_scout.enricher import enrich_papers
from arxiv_scout.scorer import score_papers

def run_pipeline(config_path: str, db_path: str = "arxiv_scout.db"):
    """Run the full fetch -> enrich -> score pipeline."""
    config = load_config(config_path)
    db = Database(db_path)

    print("=== ArXiv Scout Pipeline ===")

    # Fetch
    count = fetch_papers(db, config["categories"])
    print(f"Fetched {count} new papers")

    # Enrich
    s2_config = config.get("semantic_scholar", {})
    enrich_papers(
        db,
        config["affiliation_keywords"],
        api_key=s2_config.get("api_key") or None,
        batch_size=s2_config.get("batch_size", 500),
    )
    print("Enrichment complete")

    # Score
    score_papers(db, config)
    print("Scoring complete")

    print("=== Pipeline complete ===")
