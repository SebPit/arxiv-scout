"""Tests for CLI entry point (__main__.py)."""

from unittest.mock import patch, MagicMock
import pytest


def _run_main(args: list[str]):
    """Run main() with the given CLI arguments."""
    from arxiv_scout.__main__ import main
    with patch("sys.argv", ["arxiv-scout"] + args):
        main()


class TestDefaultCommand:
    def test_no_args_runs_pipeline_with_email(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("categories: [hep-ph]\naffiliation_keywords: [Google]\n"
                       "scoring:\n  heuristic_weight: 0.4\n  llm_weight: 0.6\n"
                       "  min_heuristic_score_for_llm: 3\n  max_papers_per_day: 10\n"
                       "anthropic:\n  model: claude-haiku-4-5-20251001\n")
        db_path = str(tmp_path / "test.db")

        with patch("arxiv_scout.pipeline.run_pipeline") as mock_pipeline:
            _run_main(["--config", str(cfg), "--db", db_path])
            mock_pipeline.assert_called_once_with(str(cfg), db_path, send_email=True)

    def test_no_args_passes_send_email_true(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("categories: [hep-ph]\naffiliation_keywords: [Google]\n"
                       "scoring:\n  heuristic_weight: 0.4\n  llm_weight: 0.6\n"
                       "  min_heuristic_score_for_llm: 3\n  max_papers_per_day: 10\n"
                       "anthropic:\n  model: claude-haiku-4-5-20251001\n")
        db_path = str(tmp_path / "test.db")

        with patch("arxiv_scout.pipeline.run_pipeline") as mock_pipeline:
            _run_main(["--config", str(cfg), "--db", db_path])
            _, kwargs = mock_pipeline.call_args
            assert kwargs["send_email"] is True


class TestRunAllCommand:
    def test_run_all_does_not_send_email(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("categories: [hep-ph]\naffiliation_keywords: [Google]\n"
                       "scoring:\n  heuristic_weight: 0.4\n  llm_weight: 0.6\n"
                       "  min_heuristic_score_for_llm: 3\n  max_papers_per_day: 10\n"
                       "anthropic:\n  model: claude-haiku-4-5-20251001\n")
        db_path = str(tmp_path / "test.db")

        with patch("arxiv_scout.pipeline.run_pipeline") as mock_pipeline:
            _run_main(["--config", str(cfg), "--db", db_path, "run-all"])
            mock_pipeline.assert_called_once_with(str(cfg), db_path)
