import os
import re
import yaml

def load_config(path: str) -> dict:
    """Load YAML config with ${ENV_VAR} substitution."""
    with open(path) as f:
        raw = f.read()
    # Substitute ${VAR_NAME} with env values (empty string if unset)
    raw = re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), ''), raw)
    return yaml.safe_load(raw)
