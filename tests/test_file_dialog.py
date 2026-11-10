"""Tests for the GUI file dialog, which runs in a child process."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from itertool.gui.file_dialog import ask_path


class TestAskPath(unittest.TestCase):
    """ask_path() runs the dialog in a child process, faked here."""

    def _ask(self, stdout: str, save: bool = False):
        fake = type("Done", (), {"stdout": stdout})()
        with patch("itertool.gui.file_dialog.subprocess.run", return_value=fake) as run:
            return ask_path("docs/a.md", [("Markdown Files", "*.md")], save=save, default_ext=".md"), run

    def test_path_inside_cwd_becomes_relative(self):
        chosen, _ = self._ask(str(Path.cwd() / "docs" / "b.md"))
        self.assertEqual(chosen, "docs/b.md")

    def test_path_outside_cwd_stays_absolute(self):
        outside = Path(Path.cwd().anchor) / "elsewhere-iter-test" / "b.md"
        chosen, _ = self._ask(str(outside))
        self.assertEqual(chosen, outside.as_posix())

    def test_cancel_returns_none(self):
        chosen, _ = self._ask("")
        self.assertIsNone(chosen)

    def test_dialog_opens_in_the_folder_of_the_current_path(self):
        _, run = self._ask("", save=True)
        args = json.loads(run.call_args.args[0][-1])
        self.assertEqual(args["initial_dir"], str(Path.cwd() / "docs"))
        self.assertTrue(args["save"])
        self.assertEqual(args["default_ext"], ".md")


if __name__ == "__main__":
    unittest.main()
