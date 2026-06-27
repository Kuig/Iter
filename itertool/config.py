"""Application configuration dataclass for Iter."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class AppConfig:
    """Top-level Iter application configuration loaded from config.json.

    Holds model settings and provider connectivity. Does NOT include
    system_prompt — that lives in each Preset so it can vary per task.

    Attributes:
        provider: AI provider name (e.g. 'ollama', 'google').
        model_name: Model identifier for main processing calls.
        embedding_provider: AI provider name for embeddings.
        embedding_model: Model identifier for embedding / cosine similarity.
        temperature: Sampling temperature for AI calls.
        ollama_url: Base URL for the Ollama server (used when provider is 'ollama').
        thinking: If True, enable extended thinking / reasoning mode where supported.
    """

    provider: str = "ollama"
    model_name: str = "gemma4:12b"
    embedding_provider: str = "ollama"
    embedding_model: str = "bge-m3"
    temperature: float = 0.7
    ollama_url: str = "http://localhost:11434"
    thinking: bool = False

    @classmethod
    def load(cls, path: str | Path = _PROJECT_ROOT / "config.json") -> AppConfig:
        """Load configuration from a JSON file with fallback defaults.

        Args:
            path: Path to the configuration file.

        Returns:
            An AppConfig instance populated from the file,
            falling back to field defaults for any missing keys.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
