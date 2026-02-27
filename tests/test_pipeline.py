from unittest.mock import patch, MagicMock
from arxiv_scout.pipeline import run_pipeline

def test_run_pipeline_calls_all_stages(tmp_db, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
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
""")
    with patch("arxiv_scout.pipeline.fetch_papers", return_value=5) as mf, \
         patch("arxiv_scout.pipeline.enrich_papers") as me, \
         patch("arxiv_scout.pipeline.score_papers") as ms:
        run_pipeline(str(cfg), db_path=tmp_db)
        mf.assert_called_once()
        me.assert_called_once()
        ms.assert_called_once()
