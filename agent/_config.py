"""Shared configuration — model resolution and constants."""
from __future__ import annotations

import os

DEFAULT_MODEL = "gpt-4.1"
LOCAL_DEFAULT_MODEL = "qwen2.5:7b"


def resolve_model(env_var: str) -> str:
    """Return the configured model, falling back to a local default for Ollama."""
    model = os.getenv(env_var, DEFAULT_MODEL)
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if "localhost:11434" in base_url and model == DEFAULT_MODEL:
        return LOCAL_DEFAULT_MODEL
    return model
