"""Tests for AppConfig loading, saving and provider registration."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import itertool.config as config_module
from itertool.config import AppConfig, resolve_config_path


class TestAppConfigLoad(unittest.TestCase):
    """load() is tolerant of unknown keys, and says so."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def _write(self, data: dict) -> Path:
        path = self.dir / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_missing_file_falls_back_to_defaults(self):
        cfg = AppConfig.load(self.dir / "absent.json")
        self.assertEqual(cfg.provider, AppConfig().provider)

    def test_known_keys_are_applied(self):
        cfg = AppConfig.load(self._write({"provider": "google", "temperature": 0.1}))
        self.assertEqual(cfg.provider, "google")
        self.assertAlmostEqual(cfg.temperature, 0.1)

    def test_unknown_keys_are_dropped_but_warned_about(self):
        path = self._write({"provider": "google", "legacy_key": 1})
        warnings: list[str] = []
        original = config_module.logger.log_warning
        config_module.logger.log_warning = warnings.append
        try:
            cfg = AppConfig.load(path)
        finally:
            config_module.logger.log_warning = original

        self.assertEqual(cfg.provider, "google")
        self.assertEqual(len(warnings), 1)
        self.assertIn("legacy_key", warnings[0])

    def test_no_warning_when_every_key_is_known(self):
        path = self._write({"provider": "google"})
        warnings: list[str] = []
        original = config_module.logger.log_warning
        config_module.logger.log_warning = warnings.append
        try:
            AppConfig.load(path)
        finally:
            config_module.logger.log_warning = original
        self.assertEqual(warnings, [])


class TestAppConfigSave(unittest.TestCase):
    """save() round-trips through load() without loss."""

    def test_round_trip(self):
        path = Path(tempfile.mkdtemp()) / "config.json"
        original = AppConfig(
            provider="google",
            model_name="gemini-3-pro",
            temperature=0.25,
            timeout=120,
            context_size=8192,
        )
        original.save(path)
        self.assertEqual(AppConfig.load(path), original)

    def test_creates_missing_parent_directory(self):
        path = Path(tempfile.mkdtemp()) / "nested" / "config.json"
        AppConfig().save(path)
        self.assertTrue(path.exists())


class TestResolveConfigPath(unittest.TestCase):
    """An explicit path always wins over the search."""

    def test_explicit_path_is_returned_unchanged(self):
        self.assertEqual(resolve_config_path("a/b.json"), Path("a/b.json"))

    def test_search_returns_a_path_named_config_json(self):
        self.assertEqual(resolve_config_path().name, "config.json")


class TestApplyToClient(unittest.TestCase):
    """Registration is idempotent, and re-applies when settings change."""

    def setUp(self):
        self.configured: list[tuple] = []
        self.preloaded: list[tuple] = []
        self._orig_configure = config_module.configure_provider
        self._orig_preload = config_module.preload_model
        self._orig_applied = config_module._applied_settings

        config_module.configure_provider = lambda name, **kw: self.configured.append((name, kw))
        config_module.preload_model = lambda p, m, **kw: self.preloaded.append((p, m, kw))
        config_module._applied_settings = None

    def tearDown(self):
        config_module.configure_provider = self._orig_configure
        config_module.preload_model = self._orig_preload
        config_module._applied_settings = self._orig_applied

    def test_first_call_registers(self):
        AppConfig().apply_to_client()
        self.assertEqual(len(self.configured), 1)
        self.assertEqual(len(self.preloaded), 1)

    def test_repeated_call_with_same_settings_is_a_no_op(self):
        cfg = AppConfig()
        cfg.apply_to_client()
        cfg.apply_to_client()
        AppConfig().apply_to_client()
        self.assertEqual(len(self.configured), 1)
        self.assertEqual(len(self.preloaded), 1)

    def test_changed_provider_re_registers(self):
        AppConfig().apply_to_client()
        AppConfig(provider="google").apply_to_client()
        self.assertEqual(len(self.configured), 2)

    def test_changed_url_re_registers(self):
        AppConfig().apply_to_client()
        AppConfig(ollama_url="http://elsewhere:11434").apply_to_client()
        self.assertEqual(len(self.configured), 2)
        self.assertEqual(self.configured[1][1]["url"], "http://elsewhere:11434")

    def test_preload_only_for_ollama(self):
        AppConfig(provider="google").apply_to_client()
        self.assertEqual(len(self.configured), 1)
        self.assertEqual(self.preloaded, [])

    def test_context_size_is_passed_to_preload(self):
        AppConfig(context_size=8192).apply_to_client()
        self.assertEqual(self.preloaded[0][2]["context_size"], 8192)

    def test_timeout_is_registered(self):
        AppConfig(timeout=42).apply_to_client()
        self.assertEqual(self.configured[0][1]["timeout"], 42)


if __name__ == "__main__":
    unittest.main()
