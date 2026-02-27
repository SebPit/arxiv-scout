"""Regression tests for scorer bugs caught in production."""

from unittest.mock import patch, MagicMock
from arxiv_scout.scorer import score_papers, parse_llm_response
from arxiv_scout.db import Database


def _make_config(threshold=3):
    return {
        "scoring": {
            "heuristic_weight": 0.4,
            "llm_weight": 0.6,
            "min_heuristic_score_for_llm": threshold,
            "max_papers_per_day": 100,
        },
        "anthropic": {"model": "test"},
    }


def test_parse_llm_response_markdown_fenced():
    """BUG: Claude wraps JSON in ```json ... ```. Parser must handle this."""
    text = '```json\n{"score": 7, "summary": "Novel approach."}\n```'
    score, summary = parse_llm_response(text)
    assert score == 7
    assert "Novel" in summary


def test_parse_llm_response_markdown_fenced_no_language():
    """Markdown fences without language tag."""
    text = '```\n{"score": 5, "summary": "OK."}\n```'
    score, summary = parse_llm_response(text)
    assert score == 5


def test_combined_score_set_even_without_llm(tmp_db):
    """BUG: Papers with heuristic_score but no LLM score must still get combined_score.

    Previously, if score_papers ran once (setting heuristic scores) and the
    Anthropic client failed, combined_score was never set. Re-running would
    then return early because get_unscored_papers() returned [] (heuristic
    already set), leaving combined_score NULL forever.
    """
    db = Database(tmp_db)
    pid = db.insert_paper(
        arxiv_id="2602.11111", title="Test", abstract="Abstract",
        categories="cs.LG", published_date="2026-02-27", arxiv_url=""
    )

    with patch("arxiv_scout.scorer.anthropic.Anthropic", side_effect=Exception("no key")):
        score_papers(db, _make_config())

    paper = db.get_paper_by_arxiv_id("2602.11111")
    assert paper["heuristic_score"] is not None
    assert paper["combined_score"] is not None, "combined_score must be set even without LLM"


def test_rerun_scoring_fills_missing_combined_scores(tmp_db):
    """BUG: If first run sets heuristic but crashes before combined,
    second run must still fill combined_score."""
    db = Database(tmp_db)
    pid = db.insert_paper(
        arxiv_id="2602.11111", title="Test", abstract="Abstract",
        categories="cs.LG", published_date="2026-02-27", arxiv_url=""
    )
    # Simulate first run that only set heuristic (crashed before combined)
    db.update_heuristic_score(pid, 5.0)

    # Second run should detect combined_score IS NULL and fix it
    with patch("arxiv_scout.scorer.anthropic.Anthropic", side_effect=Exception("no key")):
        score_papers(db, _make_config())

    paper = db.get_paper_by_arxiv_id("2602.11111")
    assert paper["combined_score"] is not None
    assert paper["combined_score"] == 5.0 * 0.4  # heuristic_weight only


def test_llm_scores_applied_above_threshold(tmp_db):
    """Papers above threshold get LLM score; combined = weighted sum."""
    db = Database(tmp_db)
    pid = db.insert_paper(
        arxiv_id="2602.11111", title="Good Paper", abstract="Novel approach",
        categories="cs.LG", published_date="2026-02-27", arxiv_url=""
    )
    # Give it a high h-index author so heuristic >= 3
    aid = db.upsert_author(name="Expert", semantic_scholar_id="e1", h_index=60, citation_count=20000)
    db.link_paper_author(pid, aid, 0)
    db.insert_affiliation(aid, "Google DeepMind", "Google")

    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"score": 8, "summary": "Excellent work."}')]
    mock_client.messages.create.return_value = mock_msg

    with patch("arxiv_scout.scorer.anthropic.Anthropic", return_value=mock_client):
        score_papers(db, _make_config(threshold=3))

    paper = db.get_paper_by_arxiv_id("2602.11111")
    assert paper["llm_score"] == 8
    assert paper["llm_summary"] == "Excellent work."
    # combined = 0.4 * heuristic + 0.6 * 8
    assert paper["combined_score"] is not None
    assert paper["combined_score"] > paper["heuristic_score"] * 0.4


def test_papers_below_threshold_get_heuristic_only_combined(tmp_db):
    """Papers below LLM threshold get combined = heuristic * h_weight."""
    db = Database(tmp_db)
    pid = db.insert_paper(
        arxiv_id="2602.11111", title="Incremental", abstract="Minor update",
        categories="cs.LG", published_date="2026-02-27", arxiv_url=""
    )
    # No affiliations, no h-index -> heuristic = 0

    mock_client = MagicMock()
    with patch("arxiv_scout.scorer.anthropic.Anthropic", return_value=mock_client):
        score_papers(db, _make_config(threshold=3))

    paper = db.get_paper_by_arxiv_id("2602.11111")
    assert paper["heuristic_score"] == 0.0
    assert paper["combined_score"] == 0.0
    # LLM should NOT have been called for this paper
    mock_client.messages.create.assert_not_called()
