"""Endpoint configuration. API keys are read from environment variables (or a local .env file) only."""
from __future__ import annotations

import os
from pathlib import Path

# Keys are intentionally empty: set OPENAI_API_KEY, DASHSCOPE_API_KEY and DEEPSEEK_API_KEY instead.
QWEN_API_KEY = ""
DEEPSEEK_API_KEY = ""
OPENAI_API_KEY = ""

DEFAULT_QWEN_BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines from a .env file without overriding existing variables."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
