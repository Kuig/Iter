"""Application configuration dataclass for Iter."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from unified_ai_client import configure_provider, warm_up

import itertool.logger as logger

_PROJECT_ROOT = Path(__file__).parent.parent

# Settings tuple most recently registered with UnifiedAiClient, so a repeated
# apply_to_client() with unchanged settings does not re-issue the network warm-up.
_applied_settings: tuple | None = None

KEEP_ALIVE_FOREVER = -1


def keep_alive_value(minutes: int) -> str | int:
    """Translate keep_alive_minutes into the value Ollama expects.

    Args:
        minutes: -1 keeps the model loaded forever, 0 unloads it after each
            request, any positive value is an idle time in minutes.

    Returns:
        -1 or 0 unchanged, otherwise an Ollama duration string such as '20m'.
    """
    if minutes <= 0:
        return minutes
    return f"{minutes}m"


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
        temperature: Sampling temperature for AI calls. None leaves it to the
            provider (for Ollama, the Modelfile).
        thinking: If True, enable extended thinking / reasoning mode where supported.
        timeout: Per-attempt request timeout in seconds, registered with every
            provider in use.
        context_size: Context window in tokens (Ollama num_ctx). None leaves the
            model default in place.
        keep_alive_minutes: How long Ollama keeps the model loaded once idle.
            -1 keeps it forever, 0 unloads it after each request.
    """

    provider: str = "ollama"
    model_name: str = "gemma4:12b"
    embedding_provider: str = "ollama"
    embedding_model: str = "bge-m3"
    temperature: float | None = None
    thinking: bool = False
    timeout: int = 600
    context_size: int | None = None
    keep_alive_minutes: int = 20

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

        Registers the timeout with every provider in use (generation and
        embedding), plus keep_alive and context_size with Ollama, then warms up
        the generation model so the first node does not pay the load latency.
        A failed warm-up is logged, not raised: the first real call reports
        the actual error.

        Idempotent: a repeated call with identical settings is a no-op, so every
        interface can call it whenever its settings become effective without
        re-issuing the warm-up request on every run.
        """
        global _applied_settings
        settings = (
            self.provider,
            self.model_name,
            self.embedding_provider,
            self.timeout,
            self.context_size,
            self.keep_alive_minutes,
        )
        if settings == _applied_settings:
            return

        for provider in sorted({self.provider, self.embedding_provider}):
            if provider == "ollama":
                # None is passed on purpose: it clears a context size registered
                # by an earlier run in the same process.
                configure_provider(
                    provider,
                    timeout=self.timeout,
                    keep_alive=keep_alive_value(self.keep_alive_minutes),
                    context_size=self.context_size,
                )
            else:
                configure_provider(provider, timeout=self.timeout)

        if not warm_up(self.provider, self.model_name):
            logger.log_warning(
                f"Warm-up of {self.provider}/{self.model_name} failed; "
                f"the first AI call will report the error."
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
