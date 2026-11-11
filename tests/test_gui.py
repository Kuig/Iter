"""Tests for the Streamlit GUI, run headless with streamlit.testing.

Provider and model listing is stubbed, so no test reaches a server.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import itertool.gui
import itertool.logger as logger

APP = str(Path(itertool.gui.__file__).parent / "app.py")


def setUpModule():
    # AppTest drives the script from the test thread, which Streamlit reports as bare
    # mode. A filter, because Streamlit resets its loggers' levels when the app starts.
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(
        lambda record: "missing ScriptRunContext" not in record.getMessage()
    )


def tearDownModule():
    # The app selects the streamlit backend on import.
    logger.set_backend("console")


class _GuiRun(unittest.TestCase):
    """Runs the app in an empty working directory holding one preset."""

    PRESET = {
        "name": "Folder",
        "node_mode": "file",
        "heading_handling": "merge_next",
        "postprocessing_scope": "corpus",
        "output_layout": "merged",
    }

    def setUp(self):
        self._cwd = os.getcwd()
        self.dir = Path(tempfile.mkdtemp())
        os.chdir(self.dir)
        (self.dir / "Presets").mkdir()
        (self.dir / "Presets" / "Folder.json").write_text(json.dumps(self.PRESET), encoding="utf-8")
        (self.dir / "docs").mkdir()
        self._patches = [
            patch("unified_ai_client.list_providers", return_value=[]),
            patch("unified_ai_client.list_models", return_value=[]),
        ]
        for p in self._patches:
            p.start()
        self.at = AppTest.from_file(APP, default_timeout=60)
        self.at.run()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        os.chdir(self._cwd)

    def _load_preset(self) -> None:
        self.at.selectbox(key="sb_preset_select").select("Folder")
        self.at.button(key="btn_load_preset").click().run()


class TestPage(_GuiRun):
    """The page renders, and the new options sit where they belong."""

    def test_page_loads_without_errors(self):
        self.assertFalse(self.at.exception)
        self.assertEqual(
            self.at.radio(key="radio_node_mode").options, ["🔹 Paragraph", "🔸 Section", "📄 File"]
        )

    def test_output_asks_for_a_folder_only_for_a_mirrored_folder(self):
        self.assertEqual(self.at.text_input(key="output_path_input").label, "Output file")
        self.at.text_input(key="input_path_input").input("docs").run()
        self.assertEqual(self.at.text_input(key="output_path_input").label, "Output folder")
        self.at.radio(key="radio_output_layout").set_value("Merged (single file)").run()
        self.assertEqual(self.at.text_input(key="output_path_input").label, "Output file")

    def test_file_mode_hides_the_handling_rules(self):
        self.at.radio(key="radio_node_mode").set_value("📄 File").run()
        self.assertFalse(any(s.key == "sel_heading_handling" for s in self.at.selectbox))


class TestPresetRoundTrip(_GuiRun):
    """The new preset fields survive load -> session state -> save."""

    def test_loaded_preset_fills_the_widgets(self):
        self._load_preset()
        self.assertEqual(self.at.radio(key="radio_node_mode").value, "📄 File")
        self.assertEqual(self.at.radio(key="radio_postprocessing_scope").value, "Whole folder")
        self.assertEqual(self.at.radio(key="radio_output_layout").value, "Merged (single file)")

    def test_saved_preset_keeps_every_new_field(self):
        self._load_preset()
        self.at.text_input(key="sb_preset_name").input("Copy").run()
        self.at.button(key="btn_save_preset").click().run()
        saved = json.loads((self.dir / "Presets" / "Copy.json").read_text(encoding="utf-8"))
        for field in ("node_mode", "heading_handling", "postprocessing_scope", "output_layout"):
            self.assertEqual(saved[field], self.PRESET[field], field)

    def test_rules_hidden_in_file_mode_come_back_with_their_value(self):
        self._load_preset()
        self.at.radio(key="radio_node_mode").set_value("🔸 Section").run()
        self.assertEqual(self.at.selectbox(key="sel_heading_handling").value, "Merge with next node")


if __name__ == "__main__":
    unittest.main()
