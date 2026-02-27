from unittest.mock import MagicMock

from arxiv_scout.scorer import compute_heuristic_score, parse_llm_response, score_with_llm


def test_heuristic_high_score():
    score = compute_heuristic_score(["Google", "DeepMind"], 80, 0)
    assert score >= 7.0


def test_heuristic_low_score():
    score = compute_heuristic_score([], 3, 0)
    assert score <= 3.0


def test_heuristic_medium_score():
    score = compute_heuristic_score(["MIT"], 20, 0)
    assert 2.0 <= score <= 7.0


def test_heuristic_bounds():
    low = compute_heuristic_score([], 0, 0)
    high = compute_heuristic_score(["A", "B", "C"], 150, 1000)
    assert 0 <= low <= 10
    assert 0 <= high <= 10


def test_parse_llm_response_valid():
    score, summary = parse_llm_response('{"score": 8, "summary": "Novel approach."}')
    assert score == 8
    assert "Novel" in summary


def test_parse_llm_response_invalid():
    score, summary = parse_llm_response("not json")
    assert score is None
    assert summary is None


def test_parse_llm_response_clamp():
    score, _ = parse_llm_response('{"score": 15, "summary": "x"}')
    assert score == 10
    score2, _ = parse_llm_response('{"score": -5, "summary": "x"}')
    assert score2 == 0


def test_score_with_llm():
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"score": 7, "summary": "Interesting method."}')]
    mock_client.messages.create.return_value = mock_msg

    score, summary = score_with_llm(mock_client, "model", "Title", "Abstract", "cs.LG")
    assert score == 7
    assert "method" in summary.lower()
    mock_client.messages.create.assert_called_once()
