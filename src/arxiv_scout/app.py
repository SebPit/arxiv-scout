from flask import Flask, render_template, request, jsonify
from arxiv_scout.db import Database
import os

def create_app(db_path: str = "arxiv_scout.db"):
    template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

    def get_db():
        return Database(db_path)

    @app.route("/")
    def index():
        db = get_db()
        category = request.args.get("category", "")
        min_score = request.args.get("min_score", type=float, default=0)
        sort = request.args.get("sort", "combined_score")
        page = request.args.get("page", type=int, default=1)
        per_page = 50

        papers = db.get_papers(
            min_score=min_score,
            category=category or None,
            order_by=sort,
            limit=per_page,
            offset=(page - 1) * per_page,
        )

        # Enrich papers with author and affiliation data
        enriched = []
        for paper in papers:
            p = dict(paper)
            authors = db.get_paper_authors(paper["id"])
            author_list = []
            for a in authors:
                affs = db.get_author_affiliations(a["id"])
                author_list.append({
                    "name": a["name"],
                    "h_index": a["h_index"],
                    "affiliations": [aff["institution_name"] for aff in affs],
                    "matched_keywords": [aff["matched_keyword"] for aff in affs if aff["matched_keyword"]],
                })
            p["author_details"] = author_list
            enriched.append(p)

        return render_template("index.html",
            papers=enriched,
            category=category,
            min_score=min_score,
            sort=sort,
            page=page,
            categories=["", "hep-ph", "cs.LG"],
        )

    @app.route("/api/papers")
    def api_papers():
        db = get_db()
        category = request.args.get("category")
        min_score = request.args.get("min_score", type=float, default=0)
        sort = request.args.get("sort", "combined_score")
        limit = request.args.get("limit", type=int, default=100)

        papers = db.get_papers(min_score=min_score, category=category, order_by=sort, limit=limit)
        return jsonify({"papers": [dict(p) for p in papers], "count": len(papers)})

    return app
