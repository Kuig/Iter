"""Tests for the CLI subcommands that need no AI call."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from itertool.__main__ import build_parser


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


if __name__ == "__main__":
    unittest.main()
