"""Application configuration dataclass for Iter."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from unified_ai_client import configure_provider, preload_model

import itertool.logger as logger

_PROJECT_ROOT = Path(__file__).parent.parent

# Settings tuple most recently registered with UnifiedAiClient, so a repeated
# apply_to_client() with unchanged settings does not re-issue the network warm-up.
_applied_settings: tuple | None = None


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Resolve which config.json a load would read.

    Args:
        path: Explicit path, or None to search.

    Returns:
        The explicit path if given, otherwise ./config.json when it exists,
        otherwise the package-root config.json (which may not exist either).
    """
    if path is not None:
        return Path(path)
    cwd_path = Path.cwd() / "config.json"
    return cwd_path if cwd_path.exists() else _PROJECT_ROOT / "config.json"


@dataclass
class AppConfig:
    """Top-level Iter application configuration loaded from config.json.

    Holds model settings and provider connectivity. Does NOT include
    system_prompt - that lives in each Preset so it can vary per task.

    Attributes:
        provider: AI provider name (e.g. 'ollama', 'google').
        model_name: Model identifier for main processing calls.
        embedding_provider: AI provider name for embeddings.
        embedding_model: Model identifier for embedding / cosine similarity.
        temperature: Sampling temperature for AI calls.
        ollama_url: Base URL for the Ollama server (used when provider is 'ollama').
        thinking: If True, enable extended thinking / reasoning mode where supported.
        timeout: Per-request timeout in seconds, registered with the provider.
        context_size: Context window in tokens (Ollama num_ctx). None leaves the
            model default in place.
    """

    provider: str = "ollama"
    model_name: str = "gemma4:12b"
    embedding_provider: str = "ollama"
    embedding_model: str = "bge-m3"
    temperature: float = 0.7
    ollama_url: str = "http://localhost:11434"
    thinking: bool = False
    timeout: int = 300
    context_size: int | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        """Load configuration from a JSON file with fallback defaults.

        Checks the current working directory first, falling back to the package root.

        Args:
            path: Optional path to the configuration file.

        Returns:
            An AppConfig instance populated from the file,
            falling back to field defaults for any missing keys.
        """
        path = resolve_config_path(path)

        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = sorted(set(data) - set(known))
        if unknown:
            logger.log_warning(f"Ignoring unknown keys in {path}: {', '.join(unknown)}")
        return cls(**known)

    def apply_to_client(self) -> None:
        """Register these settings with UnifiedAiClient, skipping if unchanged.

        Registers the Ollama base URL and timeout, and pre-loads the model when
        the provider is Ollama so the first node does not pay the load latency.

        Idempotent: a repeated call with identical settings is a no-op, so every
        interface can call it whenever its settings become effective without
        re-issuing the warm-up request on every run.
        """
        global _applied_settings
        settings = (
            self.provider,
            self.model_name,
            self.ollama_url,
            self.timeout,
            self.context_size,
        )
        if settings == _applied_settings:
            return

        # Always register the Ollama base URL so UnifiedAiClient uses the right
        # server even when the provider is not ollama (harmless no-op there).
        configure_provider("ollama", url=self.ollama_url, timeout=self.timeout)
        if self.provider == "ollama":
            preload_model(
                self.provider,
                self.model_name,
                context_size=self.context_size,
            )
        _applied_settings = settings

    def save(self, path: str | Path | None = None) -> None:
        """Write this configuration to a JSON file.

        Unlike load(), which falls back to the package root, save always targets
        the working directory: a saved configuration belongs to the workspace
        folder, never inside an installed package.

        Args:
            path: Optional destination. Defaults to ./config.json.
        """
        path = Path(path) if path is not None else Path.cwd() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
