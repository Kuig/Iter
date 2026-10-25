"""Tests for AppConfig loading, saving and provider registration."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import itertool.config as config_module
from itertool.config import AppConfig, keep_alive_value, resolve_config_path


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
        path = self._write({"provider": "google", "ollama_url": "http://localhost:11434"})
        warnings: list[str] = []
        original = config_module.logger.log_warning
        config_module.logger.log_warning = warnings.append
        try:
            cfg = AppConfig.load(path)
        finally:
            config_module.logger.log_warning = original

        self.assertEqual(cfg.provider, "google")
        self.assertEqual(len(warnings), 1)
        self.assertIn("ollama_url", warnings[0])

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
            keep_alive_minutes=-1,
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


class TestKeepAliveValue(unittest.TestCase):
    """-1 and 0 are Ollama sentinels; anything else is a duration in minutes."""

    def test_forever(self):
        self.assertEqual(keep_alive_value(-1), -1)

    def test_unload_after_each_request(self):
        self.assertEqual(keep_alive_value(0), 0)

    def test_minutes(self):
        self.assertEqual(keep_alive_value(20), "20m")


class TestApplyToClient(unittest.TestCase):
    """Registration is idempotent, and re-applies when settings change."""

    def setUp(self):
        self.configured: list[tuple] = []
        self.warmed: list[tuple] = []
        self.warm_up_result = True
        self.warnings: list[str] = []
        self._orig_configure = config_module.configure_provider
        self._orig_warm_up = config_module.warm_up
        self._orig_warning = config_module.logger.log_warning
        self._orig_applied = config_module._applied_settings

        config_module.configure_provider = lambda name, **kw: self.configured.append((name, kw))

        def fake_warm_up(provider, model, **kw):
            self.warmed.append((provider, model))
            return self.warm_up_result

        config_module.warm_up = fake_warm_up
        config_module.logger.log_warning = self.warnings.append
        config_module._applied_settings = None

    def tearDown(self):
        config_module.configure_provider = self._orig_configure
        config_module.warm_up = self._orig_warm_up
        config_module.logger.log_warning = self._orig_warning
        config_module._applied_settings = self._orig_applied

    def _registered(self) -> dict[str, dict]:
        return {name: kw for name, kw in self.configured}

    def test_first_call_registers_and_warms_up(self):
        AppConfig().apply_to_client()
        self.assertEqual(len(self.configured), 1)
        self.assertEqual(self.warmed, [("ollama", AppConfig().model_name)])

    def test_repeated_call_with_same_settings_is_a_no_op(self):
        cfg = AppConfig()
        cfg.apply_to_client()
        cfg.apply_to_client()
        AppConfig().apply_to_client()
        self.assertEqual(len(self.configured), 1)
        self.assertEqual(len(self.warmed), 1)

    def test_changed_provider_re_registers(self):
        AppConfig().apply_to_client()
        AppConfig(provider="google").apply_to_client()
        self.assertEqual(len(self.warmed), 2)

    def test_changed_keep_alive_re_registers(self):
        AppConfig().apply_to_client()
        AppConfig(keep_alive_minutes=-1).apply_to_client()
        self.assertEqual(self.configured[-1][1]["keep_alive"], -1)

    def test_timeout_is_registered_for_both_providers(self):
        AppConfig(provider="google", embedding_provider="ollama", timeout=42).apply_to_client()
        registered = self._registered()
        self.assertEqual(registered["google"]["timeout"], 42)
        self.assertEqual(registered["ollama"]["timeout"], 42)

    def test_ollama_options_are_not_sent_to_other_providers(self):
        AppConfig(provider="google", embedding_provider="google").apply_to_client()
        self.assertEqual(self._registered(), {"google": {"timeout": AppConfig().timeout}})

    def test_ollama_receives_keep_alive_and_context_size(self):
        AppConfig(context_size=8192, keep_alive_minutes=20).apply_to_client()
        ollama = self._registered()["ollama"]
        self.assertEqual(ollama["context_size"], 8192)
        self.assertEqual(ollama["keep_alive"], "20m")

    def test_unset_context_size_is_passed_to_clear_an_earlier_one(self):
        AppConfig(context_size=8192).apply_to_client()
        AppConfig(context_size=None).apply_to_client()
        self.assertIn("context_size", self.configured[-1][1])
        self.assertIsNone(self.configured[-1][1]["context_size"])

    def test_failed_warm_up_warns_instead_of_raising(self):
        self.warm_up_result = False
        AppConfig().apply_to_client()
        self.assertEqual(len(self.warnings), 1)
        self.assertIn("Warm-up", self.warnings[0])


if __name__ == "__main__":
    unittest.main()
