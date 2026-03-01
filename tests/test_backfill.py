"""Tests for the backfill feature (fetcher.backfill_papers + CLI)."""

import os
from unittest.mock import patch, MagicMock

from arxiv_scout.fetcher import backfill_papers
from arxiv_scout.db import Database

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_arxiv_api.xml")


def _mock_api_response(xml_text):
    """Create a mock requests.Response returning the given XML."""
    resp = MagicMock()
    resp.text = xml_text
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


class TestBackfillPapers:
    def test_fetches_and_inserts_papers(self, tmp_db):
        db = Database(tmp_db)
        with open(FIXTURE_PATH) as f:
            xml = f.read()
        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)), \
             patch("arxiv_scout.fetcher.time.sleep"):
            count = backfill_papers(db, ["hep-ph"], days=5)
        assert count == 3

    def test_stores_correct_metadata(self, tmp_db):
        db = Database(tmp_db)
        with open(FIXTURE_PATH) as f:
            xml = f.read()
        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)), \
             patch("arxiv_scout.fetcher.time.sleep"):
            backfill_papers(db, ["hep-ph"], days=5)

        paper = db.get_paper_by_arxiv_id("2602.11111")
        assert paper is not None
        assert paper["title"] == "Dark Matter Scattering in the Early Universe"
        assert "dark matter" in paper["abstract"].lower()
        assert paper["published_date"] == "2026-02-25"
        assert paper["arxiv_url"] == "https://arxiv.org/abs/2602.11111"

    def test_stores_authors(self, tmp_db):
        db = Database(tmp_db)
        with open(FIXTURE_PATH) as f:
            xml = f.read()
        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)), \
             patch("arxiv_scout.fetcher.time.sleep"):
            backfill_papers(db, ["hep-ph"], days=5)

        paper = db.get_paper_by_arxiv_id("2602.11111")
        authors = db.get_paper_authors(paper["id"])
        assert len(authors) == 2
        assert authors[0]["name"] == "Alice Researcher"
        assert authors[1]["name"] == "Bob Scientist"

    def test_stores_categories(self, tmp_db):
        db = Database(tmp_db)
        with open(FIXTURE_PATH) as f:
            xml = f.read()
        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)), \
             patch("arxiv_scout.fetcher.time.sleep"):
            backfill_papers(db, ["hep-ph"], days=5)

        paper = db.get_paper_by_arxiv_id("2602.33333")
        assert "hep-ph" in paper["categories"]
        assert "cs.LG" in paper["categories"]

    def test_deduplicates_on_second_run(self, tmp_db):
        db = Database(tmp_db)
        with open(FIXTURE_PATH) as f:
            xml = f.read()
        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)), \
             patch("arxiv_scout.fetcher.time.sleep"):
            count1 = backfill_papers(db, ["hep-ph"], days=5)
            count2 = backfill_papers(db, ["hep-ph"], days=5)
        assert count1 == 3
        assert count2 == 0

    def test_queries_correct_url(self, tmp_db):
        db = Database(tmp_db)
        with open(FIXTURE_PATH) as f:
            xml = f.read()
        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)) as mock_get, \
             patch("arxiv_scout.fetcher.time.sleep"):
            backfill_papers(db, ["hep-ph"], days=7)

        call_args = mock_get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert "cat:hep-ph" in params["search_query"]
        assert "submittedDate:" in params["search_query"]
        assert params["sortBy"] == "submittedDate"

    def test_multiple_categories(self, tmp_db):
        db = Database(tmp_db)
        with open(FIXTURE_PATH) as f:
            xml = f.read()
        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)) as mock_get, \
             patch("arxiv_scout.fetcher.time.sleep"):
            count = backfill_papers(db, ["hep-ph", "cs.LG"], days=5)

        # Called once per category
        assert mock_get.call_count == 2
        # First run inserts 3, second run also 3 (different query but same IDs → dedup)
        assert count == 3

    def test_cleans_multiline_titles(self, tmp_db):
        """arXiv API sometimes has newlines in titles."""
        db = Database(tmp_db)
        with open(FIXTURE_PATH) as f:
            xml = f.read()
        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)), \
             patch("arxiv_scout.fetcher.time.sleep"):
            backfill_papers(db, ["hep-ph"], days=5)

        # The fixture has a title with newline: "Dark Matter Scattering in\n  the Early Universe"
        paper = db.get_paper_by_arxiv_id("2602.11111")
        assert "\n" not in paper["title"]
        assert paper["title"] == "Dark Matter Scattering in the Early Universe"


class TestBackfillCLI:
    def _run_main(self, args):
        from arxiv_scout.__main__ import main
        with patch("sys.argv", ["arxiv-scout"] + args):
            main()

    def test_backfill_command_runs_full_pipeline(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "categories: [hep-ph]\naffiliation_keywords: [Google]\n"
            "scoring:\n  heuristic_weight: 0.4\n  llm_weight: 0.6\n"
            "  min_heuristic_score_for_llm: 3\n  max_papers_per_day: 10\n"
            "anthropic:\n  model: claude-haiku-4-5-20251001\n"
        )
        db_path = str(tmp_path / "test.db")

        with open(FIXTURE_PATH) as f:
            xml = f.read()

        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)), \
             patch("arxiv_scout.fetcher.time.sleep"), \
             patch("arxiv_scout.enricher.enrich_papers") as mock_enrich, \
             patch("arxiv_scout.scorer.score_papers") as mock_score:
            self._run_main(["--config", str(cfg), "--db", db_path, "backfill", "--days", "5"])
            mock_enrich.assert_called_once()
            mock_score.assert_called_once()

    def test_backfill_single_category_flag(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "categories: [hep-ph, cs.LG]\naffiliation_keywords: [Google]\n"
            "scoring:\n  heuristic_weight: 0.4\n  llm_weight: 0.6\n"
            "  min_heuristic_score_for_llm: 3\n  max_papers_per_day: 10\n"
            "anthropic:\n  model: claude-haiku-4-5-20251001\n"
        )
        db_path = str(tmp_path / "test.db")

        with open(FIXTURE_PATH) as f:
            xml = f.read()

        with patch("arxiv_scout.fetcher.requests.get", return_value=_mock_api_response(xml)) as mock_get, \
             patch("arxiv_scout.fetcher.time.sleep"), \
             patch("arxiv_scout.enricher.enrich_papers"), \
             patch("arxiv_scout.scorer.score_papers"):
            self._run_main(["--config", str(cfg), "--db", db_path, "backfill", "--days", "3", "--category", "cs.LG"])
            # Should only query cs.LG, not hep-ph
            params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1].get("params")
            assert "cat:cs.LG" in params["search_query"]
