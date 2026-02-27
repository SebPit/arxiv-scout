from unittest.mock import patch, MagicMock
from arxiv_scout.pipeline import run_pipeline

SAMPLE_CONFIG = """
categories: [hep-ph]
affiliation_keywords: [Google]
scoring:
  heuristic_weight: 0.4
  llm_weight: 0.6
  min_heuristic_score_for_llm: 3
  max_papers_per_day: 10
anthropic:
  model: claude-haiku-4-5-20251001
server:
  port: 5000
  host: 127.0.0.1
"""

def test_run_pipeline_calls_all_stages(tmp_db, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(SAMPLE_CONFIG)
    with patch("arxiv_scout.pipeline.fetch_papers", return_value=5) as mf, \
         patch("arxiv_scout.pipeline.enrich_papers") as me, \
         patch("arxiv_scout.pipeline.score_papers") as ms:
        run_pipeline(str(cfg), db_path=tmp_db)
        mf.assert_called_once()
        me.assert_called_once()
        ms.assert_called_once()

def test_pipeline_send_email_when_enabled(tmp_db, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(SAMPLE_CONFIG + "\nemail:\n  enabled: true\n")
    with patch("arxiv_scout.pipeline.fetch_papers", return_value=5), \
         patch("arxiv_scout.pipeline.enrich_papers"), \
         patch("arxiv_scout.pipeline.score_papers"), \
         patch("arxiv_scout.emailer.send_digest") as mock_digest:
        run_pipeline(str(cfg), db_path=tmp_db, send_email=True)
        mock_digest.assert_called_once()

def test_pipeline_no_email_when_disabled(tmp_db, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(SAMPLE_CONFIG + "\nemail:\n  enabled: false\n")
    with patch("arxiv_scout.pipeline.fetch_papers", return_value=5), \
         patch("arxiv_scout.pipeline.enrich_papers"), \
         patch("arxiv_scout.pipeline.score_papers"), \
         patch("arxiv_scout.emailer.send_digest") as mock_digest:
        run_pipeline(str(cfg), db_path=tmp_db, send_email=True)
        mock_digest.assert_not_called()

def test_pipeline_no_email_by_default(tmp_db, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(SAMPLE_CONFIG + "\nemail:\n  enabled: true\n")
    with patch("arxiv_scout.pipeline.fetch_papers", return_value=5), \
         patch("arxiv_scout.pipeline.enrich_papers"), \
         patch("arxiv_scout.pipeline.score_papers"), \
         patch("arxiv_scout.emailer.send_digest") as mock_digest:
        run_pipeline(str(cfg), db_path=tmp_db)  # send_email defaults to False
        mock_digest.assert_not_called()
