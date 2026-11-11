"""Tests for the CLI subcommands that need no AI call."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from itertool.__main__ import build_parser
from itertool.config import AppConfig


class TestGuiCommand(unittest.TestCase):
    """iter gui launches Streamlit, forwarding the input file to the script."""

    def _command(self, *argv: str) -> list[str]:
        args = build_parser().parse_args(["gui", *argv])
        with patch("itertool.__main__.subprocess.run") as run:
            args.func(args)
        return run.call_args.args[0]

    def test_without_input_passes_no_script_arguments(self):
        self.assertNotIn("--", self._command())

    def test_input_is_forwarded_as_absolute_path(self):
        command = self._command("doc.md")
        self.assertEqual(command[-3:], ["--", "--input", str(Path("doc.md").resolve())])


class TestRunCommand(unittest.TestCase):
    """iter run passes the input, output and context to run_iter."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.dir = Path(tempfile.mkdtemp())
        os.chdir(self.dir)
        (self.dir / "docs").mkdir()
        (self.dir / "docs" / "a.md").write_text("Alpha.\n", encoding="utf-8")
        presets = self.dir / "Presets"
        presets.mkdir()
        (presets / "Merge.json").write_text('{"output_layout": "merged"}', encoding="utf-8")

    def tearDown(self):
        os.chdir(self._cwd)

    def _run_kwargs(self, *argv: str) -> dict:
        args = build_parser().parse_args(["run", *argv])
        with patch("itertool.__main__.run_iter") as run_iter, \
                patch.object(AppConfig, "load", return_value=AppConfig()), \
                patch.object(AppConfig, "apply_to_client"):
            args.func(args)
        return run_iter.call_args.kwargs

    def test_explicit_output_and_context_are_forwarded(self):
        kwargs = self._run_kwargs("--input", "docs", "--output", "out", "--context", "glossary.md")
        self.assertEqual(kwargs["input_path"], "docs")
        self.assertEqual(kwargs["output_file"], "out")
        self.assertEqual(kwargs["context_file"], "glossary.md")

    def test_default_output_of_a_file(self):
        kwargs = self._run_kwargs("--input", str(Path("docs") / "a.md"))
        self.assertEqual(kwargs["output_file"], str(Path("docs") / "a_processed.md"))
        self.assertIsNone(kwargs["context_file"])

    def test_default_output_of_a_folder_follows_the_preset_layout(self):
        mirror = self._run_kwargs("--input", "docs")
        merged = self._run_kwargs("--input", "docs", "--preset", "Merge")
        self.assertEqual(mirror["output_file"], str(self.dir.resolve() / "docs_processed"))
        self.assertEqual(merged["output_file"], str(self.dir.resolve() / "docs_processed.md"))

    def test_file_node_mode_is_accepted(self):
        kwargs = self._run_kwargs("--input", "docs", "--node-mode", "file")
        self.assertEqual(kwargs["preset"].node_mode, "file")


if __name__ == "__main__":
    unittest.main()
