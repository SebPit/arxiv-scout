"""Tests for the email digest module."""

from unittest.mock import patch, MagicMock, call
from datetime import datetime, timezone

import pytest

from arxiv_scout.emailer import build_digest_html, send_email, send_digest
from arxiv_scout.db import Database


@pytest.fixture
def db_with_papers(tmp_db):
    """Create a database with scored papers for digest testing."""
    db = Database(tmp_db)
    # Insert three papers with different scores
    p1 = db.insert_paper(
        arxiv_id="2401.00001",
        title="High-Score Paper on Transformers",
        abstract="A groundbreaking paper about transformer architectures for particle physics.",
        categories="hep-ph cs.LG",
        published_date="2024-01-15",
        arxiv_url="https://arxiv.org/abs/2401.00001",
    )
    p2 = db.insert_paper(
        arxiv_id="2401.00002",
        title="Medium-Score Paper on GANs",
        abstract="An interesting paper about generative adversarial networks.",
        categories="cs.LG",
        published_date="2024-01-15",
        arxiv_url="https://arxiv.org/abs/2401.00002",
    )
    p3 = db.insert_paper(
        arxiv_id="2401.00003",
        title="Low-Score Paper",
        abstract="A routine paper.",
        categories="hep-ph",
        published_date="2024-01-15",
        arxiv_url="https://arxiv.org/abs/2401.00003",
    )

    # Set scores using integer paper IDs
    db.update_heuristic_score(p1, 8.0)
    db.update_llm_score(p1, 9, "Revolutionary transformer architecture")
    db.update_combined_score(p1, 8.7)

    db.update_heuristic_score(p2, 5.0)
    db.update_llm_score(p2, 6, "Solid GAN work")
    db.update_combined_score(p2, 5.7)

    db.update_heuristic_score(p3, 2.0)
    db.update_combined_score(p3, 2.0)

    # Add authors with affiliations for paper 1
    auth1_id = db.upsert_author("Alice Researcher", "s2_auth1", 50, 1200)
    db.link_paper_author(p1, auth1_id, position=0)
    db.insert_affiliation(auth1_id, "Google DeepMind", matched_keyword="DeepMind")

    auth2_id = db.upsert_author("Bob Scientist", "s2_auth2", 10, 100)
    db.link_paper_author(p1, auth2_id, position=1)

    return db


class TestBuildDigestHtml:
    def test_returns_html_string(self, db_with_papers):
        html = build_digest_html(db_with_papers)
        assert "<html>" in html
        assert "ArXiv Scout Daily Digest" in html

    def test_contains_paper_titles(self, db_with_papers):
        html = build_digest_html(db_with_papers)
        assert "High-Score Paper on Transformers" in html
        assert "Medium-Score Paper on GANs" in html

    def test_contains_arxiv_links(self, db_with_papers):
        html = build_digest_html(db_with_papers)
        assert "https://arxiv.org/abs/2401.00001" in html

    def test_contains_scores(self, db_with_papers):
        html = build_digest_html(db_with_papers)
        assert "8.7" in html  # combined score
        assert "5.7" in html

    def test_contains_author_names(self, db_with_papers):
        html = build_digest_html(db_with_papers)
        assert "Alice Researcher" in html

    def test_contains_affiliation_badges(self, db_with_papers):
        html = build_digest_html(db_with_papers)
        assert "DeepMind" in html

    def test_contains_llm_summary(self, db_with_papers):
        html = build_digest_html(db_with_papers)
        assert "Revolutionary transformer architecture" in html

    def test_top_n_limits_results(self, db_with_papers):
        html = build_digest_html(db_with_papers, top_n=1)
        assert "High-Score Paper on Transformers" in html
        assert "Medium-Score Paper on GANs" not in html

    def test_abstract_truncation(self, db_with_papers):
        html = build_digest_html(db_with_papers)
        # Abstracts are short enough not to be truncated here
        assert "groundbreaking paper" in html

    def test_score_badge_colors(self, db_with_papers):
        html = build_digest_html(db_with_papers)
        # High score (>=7) should be green
        assert "#28a745" in html
        # Medium score (>=4) should be yellow
        assert "#ffc107" in html

    def test_empty_database(self, tmp_db):
        db = Database(tmp_db)
        html = build_digest_html(db)
        assert "<html>" in html
        assert "ArXiv Scout Daily Digest" in html
        assert "0 papers tracked" in html


class TestSendEmail:
    def test_sends_via_smtp(self):
        smtp_config = {
            "sender": "test@gmail.com",
            "recipient": "dest@gmail.com",
            "password": "app-password",
            "host": "smtp.gmail.com",
            "port": 587,
        }
        with patch("arxiv_scout.emailer.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            send_email("Test Subject", "<html>body</html>", smtp_config)

            mock_smtp.assert_called_once_with("smtp.gmail.com", 587, timeout=30)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("test@gmail.com", "app-password")
            mock_server.sendmail.assert_called_once()

    def test_recipient_defaults_to_sender(self):
        smtp_config = {
            "sender": "me@gmail.com",
            "password": "pass",
            "host": "smtp.gmail.com",
            "port": 587,
        }
        with patch("arxiv_scout.emailer.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            send_email("Subject", "<html>body</html>", smtp_config)

            # Check sendmail was called with sender as recipient
            sendmail_call = mock_server.sendmail.call_args
            assert sendmail_call[0][1] == ["me@gmail.com"]


class TestSendDigest:
    def test_dry_run_prints_html(self, db_with_papers, capsys):
        config = {}
        send_digest(db_with_papers, config, dry_run=True)
        captured = capsys.readouterr()
        assert "ArXiv Scout Daily Digest" in captured.out
        assert "dry run" in captured.out

    def test_missing_credentials_prints_error(self, db_with_papers, capsys):
        config = {"email": {"sender": "", "password": ""}}
        with patch.dict("os.environ", {}, clear=True):
            send_digest(db_with_papers, config)
        captured = capsys.readouterr()
        assert "SMTP_EMAIL" in captured.out

    def test_sends_when_configured(self, db_with_papers):
        config = {
            "email": {
                "sender": "scout@gmail.com",
                "password": "app-pass",
                "host": "smtp.gmail.com",
                "port": 587,
            }
        }
        env = {
            "SMTP_EMAIL": "scout@gmail.com",
            "SMTP_PASSWORD": "app-pass",
            "SMTP_RECIPIENT": "user@gmail.com",
        }
        with patch.dict("os.environ", env), \
             patch("arxiv_scout.emailer.send_email") as mock_send:
            send_digest(db_with_papers, config)
            mock_send.assert_called_once()
            subject = mock_send.call_args[0][0]
            assert "ArXiv Scout Digest" in subject

    def test_env_vars_override_config(self, db_with_papers):
        config = {
            "email": {
                "sender": "config@gmail.com",
                "password": "config-pass",
            }
        }
        env = {
            "SMTP_EMAIL": "env@gmail.com",
            "SMTP_PASSWORD": "env-pass",
            "SMTP_RECIPIENT": "envrecip@gmail.com",
        }
        with patch.dict("os.environ", env), \
             patch("arxiv_scout.emailer.send_email") as mock_send:
            send_digest(db_with_papers, config)
            smtp_config = mock_send.call_args[0][2]
            assert smtp_config["sender"] == "env@gmail.com"
            assert smtp_config["password"] == "env-pass"
            assert smtp_config["recipient"] == "envrecip@gmail.com"

    def test_top_n_forwarded(self, db_with_papers):
        config = {}
        send_digest(db_with_papers, config, top_n=5, dry_run=True)
        # Just ensure it doesn't crash with custom top_n

    def test_auth_failure_prints_helpful_error(self, db_with_papers, capsys):
        import smtplib
        config = {"email": {"host": "smtp.gmail.com", "port": 587}}
        env = {
            "SMTP_EMAIL": "test@gmail.com",
            "SMTP_PASSWORD": "wrong-password",
            "SMTP_RECIPIENT": "dest@gmail.com",
        }
        with patch.dict("os.environ", env), \
             patch("arxiv_scout.emailer.send_email",
                   side_effect=smtplib.SMTPAuthenticationError(535, b"BadCredentials")):
            send_digest(db_with_papers, config)
        captured = capsys.readouterr()
        assert "App Password" in captured.out
        assert "apppasswords" in captured.out

    def test_smtp_generic_error_handled(self, db_with_papers, capsys):
        import smtplib
        config = {"email": {"host": "smtp.gmail.com", "port": 587}}
        env = {
            "SMTP_EMAIL": "test@gmail.com",
            "SMTP_PASSWORD": "pass",
            "SMTP_RECIPIENT": "dest@gmail.com",
        }
        with patch.dict("os.environ", env), \
             patch("arxiv_scout.emailer.send_email",
                   side_effect=smtplib.SMTPException("Connection refused")):
            send_digest(db_with_papers, config)
        captured = capsys.readouterr()
        assert "Error sending email" in captured.out
