"""CLI entry point: python -m arxiv_scout"""
import argparse
import os
import sys

# Auto-load .env file (looks in cwd and parent dirs)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(prog="arxiv-scout", description="Discover influential arXiv papers")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--db", default="arxiv_scout.db", help="Path to SQLite database")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("fetch", help="Fetch new papers from arXiv RSS")
    sub.add_parser("enrich", help="Enrich papers with Semantic Scholar data")
    sub.add_parser("score", help="Score papers (heuristic + LLM)")
    sub.add_parser("run-all", help="Run full pipeline (fetch + enrich + score)")

    serve_parser = sub.add_parser("serve", help="Start web dashboard")
    serve_parser.add_argument("--port", type=int, default=None, help="Override port")
    serve_parser.add_argument("--no-scheduler", action="store_true", help="Disable scheduled fetching")

    email_parser = sub.add_parser("email", help="Send email digest of top papers")
    email_parser.add_argument("--top", type=int, default=20, help="Number of top papers to include")
    email_parser.add_argument("--dry-run", action="store_true", help="Print email HTML instead of sending")

    args = parser.parse_args()

    if args.command is None:
        # Default: run full pipeline + send email digest
        from arxiv_scout.pipeline import run_pipeline
        run_pipeline(args.config, args.db, send_email=True)
        return

    from arxiv_scout.config import load_config
    from arxiv_scout.db import Database

    config = load_config(args.config)

    if args.command == "fetch":
        from arxiv_scout.fetcher import fetch_papers
        db = Database(args.db)
        count = fetch_papers(db, config["categories"])
        print(f"Fetched {count} new papers")

    elif args.command == "enrich":
        from arxiv_scout.enricher import enrich_papers
        db = Database(args.db)
        s2 = config.get("semantic_scholar", {})
        enrich_papers(db, config["affiliation_keywords"], api_key=s2.get("api_key"), batch_size=s2.get("batch_size", 500))
        print("Enrichment complete")

    elif args.command == "score":
        from arxiv_scout.scorer import score_papers
        db = Database(args.db)
        score_papers(db, config)
        print("Scoring complete")

    elif args.command == "run-all":
        from arxiv_scout.pipeline import run_pipeline
        run_pipeline(args.config, args.db)

    elif args.command == "email":
        from arxiv_scout.emailer import send_digest
        db = Database(args.db)
        send_digest(db, config, top_n=args.top, dry_run=args.dry_run)

    elif args.command == "serve":
        from arxiv_scout.app import create_app
        port = args.port or config.get("server", {}).get("port", 5000)
        host = config.get("server", {}).get("host", "127.0.0.1")

        if not args.no_scheduler:
            from arxiv_scout.scheduler import start_scheduler
            fetch_time = config.get("schedule", {}).get("fetch_time", "06:00")
            scheduler = start_scheduler(args.config, args.db, fetch_time)
            print(f"Scheduler started (daily at {fetch_time})")

        app = create_app(db_path=args.db)
        print(f"Dashboard at http://{host}:{port}")
        app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
