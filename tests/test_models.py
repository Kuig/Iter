"""Tests for Preset persistence and the similarity helpers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import itertool.models as models
from itertool.models import Preset, cosine_similarity


class TestPresetRoundTrip(unittest.TestCase):
    """save() then load() must return an equal Preset."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_round_trip_preserves_every_field(self):
        original = Preset(
            name="Round trip",
            system_prompt="Be terse.",
            node_mode="section",
            node_prompt="Condense.",
            postprocessing_prompt="Polish.",
            heading_handling="merge_next",
            non_text_handling="part_of_section",
            compute_similarity=True,
        )
        path = self.dir / "preset.json"
        original.save(path)
        self.assertEqual(Preset.load(path), original)

    def test_save_creates_missing_parent_directory(self):
        path = self.dir / "nested" / "preset.json"
        Preset().save(path)
        self.assertTrue(path.exists())

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            Preset.load(self.dir / "absent.json")


class TestPresetUnknownKeys(unittest.TestCase):
    """Old preset files stay loadable, but the loss is reported."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def _write(self, data: dict) -> Path:
        path = self.dir / "preset.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _load_capturing_warnings(self, path: Path):
        warnings: list[str] = []
        original = models.logger.log_warning
        models.logger.log_warning = warnings.append
        try:
            return Preset.load(path), warnings
        finally:
            models.logger.log_warning = original

    def test_unknown_key_is_dropped_but_warned_about(self):
        path = self._write({"node_prompt": "Condense.", "retired_field": True})
        preset, warnings = self._load_capturing_warnings(path)
        self.assertEqual(preset.node_prompt, "Condense.")
        self.assertEqual(len(warnings), 1)
        self.assertIn("retired_field", warnings[0])

    def test_no_warning_when_every_key_is_known(self):
        path = self._write({"node_prompt": "Condense."})
        _, warnings = self._load_capturing_warnings(path)
        self.assertEqual(warnings, [])

    def test_constructor_still_rejects_unknown_kwargs(self):
        with self.assertRaises(TypeError):
            Preset(retired_field=True)


class TestCosineSimilarity(unittest.TestCase):
    """The similarity helper must not fabricate a score from empty vectors."""

    def test_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_empty_vector_yields_zero(self):
        self.assertEqual(cosine_similarity([], [1.0, 0.0]), 0.0)


class TestShippedPresets(unittest.TestCase):
    """Every preset shipped in Presets/ must still load."""

    def test_all_shipped_presets_load(self):
        preset_dir = Path(__file__).resolve().parent.parent / "Presets"
        files = sorted(preset_dir.glob("*.json"))
        self.assertTrue(files, "no presets found to check")
        for path in files:
            with self.subTest(preset=path.name):
                self.assertIsInstance(Preset.load(path), Preset)


if __name__ == "__main__":
    unittest.main()
