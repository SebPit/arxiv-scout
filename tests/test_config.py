from arxiv_scout.config import load_config

def test_load_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
categories:
  - hep-ph
affiliation_keywords:
  - Google
scoring:
  heuristic_weight: 0.4
  llm_weight: 0.6
  min_heuristic_score_for_llm: 3
  max_papers_per_day: 100
anthropic:
  model: claude-haiku-4-5-20251001
server:
  port: 5000
  host: 127.0.0.1
""")
    config = load_config(str(cfg_file))
    assert config["categories"] == ["hep-ph"]
    assert "Google" in config["affiliation_keywords"]
    assert config["scoring"]["heuristic_weight"] == 0.4

def test_load_config_env_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key-123")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
categories: [hep-ph]
affiliation_keywords: []
scoring:
  heuristic_weight: 0.4
  llm_weight: 0.6
  min_heuristic_score_for_llm: 3
  max_papers_per_day: 100
anthropic:
  model: claude-haiku-4-5-20251001
semantic_scholar:
  api_key: ${SEMANTIC_SCHOLAR_API_KEY}
server:
  port: 5000
  host: 127.0.0.1
""")
    config = load_config(str(cfg_file))
    assert config["semantic_scholar"]["api_key"] == "test-key-123"
