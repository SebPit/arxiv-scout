"""Extensive scorer tests covering edge cases."""

from unittest.mock import patch, MagicMock
from arxiv_scout.scorer import (
    compute_heuristic_score,
    parse_llm_response,
    score_with_llm,
    score_papers,
)
from arxiv_scout.db import Database


# --- Heuristic scoring edge cases ---

def test_heuristic_zero_everything():
    score = compute_heuristic_score([], 0, 0)
    assert score == 0.0


def test_heuristic_only_affiliations():
    score = compute_heuristic_score(["Google"], 0, 0)
    assert score == 2.0  # 1 keyword * 2.0 = 2.0


def test_heuristic_affiliations_cap_at_4():
    score = compute_heuristic_score(["A", "B", "C", "D", "E"], 0, 0)
    assert score == 4.0  # capped at 4


def test_heuristic_only_h_index():
    score = compute_heuristic_score([], 30, 0)
    assert score == 2.0  # 30/60 * 4 = 2.0


def test_heuristic_h_index_cap_at_4():
    score = compute_heuristic_score([], 120, 0)
    assert score == 4.0  # capped at 60 -> 4.0 max


def test_heuristic_only_citations():
    score = compute_heuristic_score([], 0, 100)
    # log1p(100)/log1p(100) * 2 = 2.0
    assert abs(score - 2.0) < 0.01


def test_heuristic_none_h_index():
    score = compute_heuristic_score(["MIT"], None, 0)
    assert score == 2.0  # affiliations only, h_index treated as 0


def test_heuristic_max_everything():
    score = compute_heuristic_score(["A", "B", "C"], 100, 10000)
    assert score == 10.0


# --- LLM response parsing ---

def test_parse_llm_response_with_extra_fields():
    text = '{"score": 5, "summary": "OK paper.", "confidence": "high"}'
    score, summary = parse_llm_response(text)
    assert score == 5
    assert summary == "OK paper."


def test_parse_llm_response_missing_summary():
    text = '{"score": 5}'
    score, summary = parse_llm_response(text)
    assert score == 5
    assert summary == ""  # defaults to empty string


def test_parse_llm_response_missing_score():
    text = '{"summary": "No score here."}'
    score, summary = parse_llm_response(text)
    assert score is None


def test_parse_llm_response_float_score():
    text = '{"score": 7.5, "summary": "Good."}'
    score, summary = parse_llm_response(text)
    assert score == 7  # int() truncates


def test_parse_llm_response_negative():
    text = '{"score": -3, "summary": "Bad."}'
    score, _ = parse_llm_response(text)
    assert score == 0


def test_parse_llm_response_string_score():
    text = '{"score": "eight", "summary": "Bad."}'
    score, _ = parse_llm_response(text)
    assert score is None


def test_parse_llm_response_empty_string():
    score, summary = parse_llm_response("")
    assert score is None


# --- score_with_llm ---

def test_score_with_llm_empty_abstract():
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"score": 3, "summary": "No abstract provided."}')]
    mock_client.messages.create.return_value = mock_msg

    score, summary = score_with_llm(mock_client, "model", "Title Only", "", "hep-ph")
    assert score == 3


# --- score_papers orchestration ---

def test_score_papers_no_unscored(tmp_db):
    """score_papers does nothing when all papers are already scored."""
    db = Database(tmp_db)
    pid = db.insert_paper(arxiv_id="2602.11111", title="A", abstract="", categories="", published_date="", arxiv_url="")
    db.update_heuristic_score(pid, 5.0)

    config = {"scoring": {"heuristic_weight": 0.4, "llm_weight": 0.6, "min_heuristic_score_for_llm": 3, "max_papers_per_day": 10}, "anthropic": {"model": "test"}}
    # Should not raise
    score_papers(db, config)


def test_score_papers_below_threshold_no_llm(tmp_db):
    """Papers below threshold get heuristic-only combined score."""
    db = Database(tmp_db)
    pid = db.insert_paper(arxiv_id="2602.11111", title="A", abstract="Low quality", categories="", published_date="", arxiv_url="")
    # No affiliations, no h-index -> heuristic will be 0

    config = {
        "scoring": {"heuristic_weight": 0.4, "llm_weight": 0.6, "min_heuristic_score_for_llm": 3, "max_papers_per_day": 10},
        "anthropic": {"model": "test"}
    }

    with patch("arxiv_scout.scorer.anthropic.Anthropic", side_effect=Exception("no key")):
        score_papers(db, config)

    paper = db.get_paper_by_arxiv_id("2602.11111")
    assert paper["heuristic_score"] == 0.0
    assert paper["combined_score"] is not None


def test_score_papers_respects_max_llm_cap(tmp_db):
    """LLM scoring stops after max_papers_per_day."""
    db = Database(tmp_db)
    # Create 5 papers with high heuristic (affiliate match)
    for i in range(5):
        pid = db.insert_paper(
            arxiv_id=f"2602.{10000+i}", title=f"Paper {i}",
            abstract="Test", categories="", published_date="", arxiv_url=""
        )
        aid = db.upsert_author(name=f"Author {i}", semantic_scholar_id=f"s{i}", h_index=80, citation_count=50000)
        db.link_paper_author(pid, aid, 0)
        db.insert_affiliation(aid, "Google DeepMind", "Google")

    config = {
        "scoring": {"heuristic_weight": 0.4, "llm_weight": 0.6, "min_heuristic_score_for_llm": 3, "max_papers_per_day": 2},
        "anthropic": {"model": "test"}
    }

    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"score": 8, "summary": "Good."}')]
    mock_client.messages.create.return_value = mock_msg

    with patch("arxiv_scout.scorer.anthropic.Anthropic", return_value=mock_client):
        score_papers(db, config)

    # Only 2 papers should have been LLM-scored
    assert mock_client.messages.create.call_count == 2
