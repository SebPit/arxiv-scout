import json
import math
import re
from datetime import datetime, timezone

import anthropic


def compute_heuristic_score(
    matched_keywords: list[str], max_h_index: int | None, citation_count: int
) -> float:
    """Score 0-10 based on institutional affiliations, author h-index, citations."""
    # Affiliation: 0-4 points (each unique keyword match = 2pts, capped at 4)
    affiliation_score = min(4.0, len(matched_keywords) * 2.0)

    # h-index: 0-4 points (linear scaling, saturates at h=60)
    h_score = min(4.0, ((max_h_index or 0) / 60.0) * 4.0)

    # Citations: 0-2 points (log scale, mostly 0 for new papers)
    cite_score = (
        min(2.0, math.log1p(citation_count) / math.log1p(100) * 2.0)
        if citation_count
        else 0.0
    )

    return round(min(10.0, max(0.0, affiliation_score + h_score + cite_score)), 2)


def parse_llm_response(text: str) -> tuple:
    """Parse LLM JSON response. Returns (score, summary) or (None, None).

    Handles both raw JSON and markdown-fenced JSON (```json ... ```).
    """
    try:
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            # Remove closing fence
            cleaned = re.sub(r'\n?```$', '', cleaned)
            cleaned = cleaned.strip()
        data = json.loads(cleaned)
        score = int(data["score"])
        score = max(0, min(10, score))  # clamp
        summary = str(data.get("summary", ""))
        return score, summary
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None, None


def _build_scoring_prompt(interests: dict | None) -> str:
    """Build the LLM system prompt, optionally including research interests."""
    base = (
        "You are a research paper evaluator. Rate this paper's novelty and "
        'potential impact. Respond with JSON only: {"score": <1-10>, "summary": '
        '"<2-3 sentence assessment>"}\n'
        "Score guide: 1-3=incremental, 4-6=solid contribution, 7-8=significant, "
        "9-10=potentially groundbreaking"
    )
    if not interests:
        return base

    parts = [base, "\nYour evaluation should reflect these research preferences:"]
    boost = interests.get("boost", [])
    penalize = interests.get("penalize", [])
    if boost:
        parts.append("BOOST (score higher) papers about: " + "; ".join(boost))
    if penalize:
        parts.append("PENALIZE (score lower) papers about: " + "; ".join(penalize))
    return "\n".join(parts)


def score_with_llm(
    client, model: str, title: str, abstract: str, categories: str,
    interests: dict | None = None,
) -> tuple:
    """Score a paper using Claude. Returns (score, summary)."""
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=_build_scoring_prompt(interests),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Title: {title}\nCategories: {categories}\nAbstract: {abstract}"
                ),
            }
        ],
    )
    return parse_llm_response(message.content[0].text)


def score_papers(db, config: dict):
    """Orchestrate scoring: heuristic for all, LLM for top candidates."""
    weights = config.get("scoring", {})
    h_weight = weights.get("heuristic_weight", 0.4)
    l_weight = weights.get("llm_weight", 0.6)
    threshold = weights.get("min_heuristic_score_for_llm", 3)
    max_llm = weights.get("max_papers_per_day", 100)

    # Phase 1: Heuristic scoring for papers that don't have one yet
    unscored = db.get_unscored_papers()
    for paper in unscored:
        authors = db.get_paper_authors(paper["id"])
        all_keywords = set()
        max_h = 0
        for author in authors:
            affs = db.get_author_affiliations(author["id"])
            for aff in affs:
                if aff["matched_keyword"]:
                    all_keywords.add(aff["matched_keyword"])
            max_h = max(max_h, author.get("h_index") or 0)

        h_score = compute_heuristic_score(
            list(all_keywords), max_h, paper["citation_count"] or 0
        )
        db.update_heuristic_score(paper["id"], h_score)

    # Phase 2: LLM scoring + combined scores for papers missing combined_score
    needs_combined = db.execute(
        "SELECT * FROM papers WHERE combined_score IS NULL AND heuristic_score IS NOT NULL"
    ).fetchall()
    needs_combined = [dict(row) for row in needs_combined]

    if not needs_combined:
        return

    # Try to create Anthropic client for LLM scoring
    model = config.get("anthropic", {}).get("model", "claude-haiku-4-5-20251001")
    interests = config.get("research_interests")
    client = None
    try:
        client = anthropic.Anthropic()
    except Exception:
        print("Warning: Could not create Anthropic client. Skipping LLM scoring.")

    llm_count = 0
    for paper in needs_combined:
        h_score = paper["heuristic_score"]
        # Papers attempted on Semantic Scholar but not found have an
        # artificially low heuristic (no author metadata).  Always send
        # them to the LLM so they get a fair score based on title + abstract.
        s2_attempted_not_found = (
            paper["enrichment_attempted_at"] is not None
            and paper["enrichment_found"] == 0
        )
        eligible = h_score >= threshold or s2_attempted_not_found

        if client and eligible and llm_count < max_llm:
            llm_score, summary = score_with_llm(
                client, model, paper["title"], paper["abstract"] or "", paper["categories"] or "",
                interests=interests,
            )
            if llm_score is not None:
                db.update_llm_score(paper["id"], llm_score, summary)
                combined = h_weight * h_score + l_weight * llm_score
                db.update_combined_score(paper["id"], round(combined, 2))
                llm_count += 1
                continue

        # No LLM score — combined is heuristic-weighted only
        db.update_combined_score(paper["id"], round(h_score * h_weight, 2))
