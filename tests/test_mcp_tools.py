"""Tests for the MCP tools, registered on a stand-in for FastMCP."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import itertool.logger as logger
import itertool.mcp_tools as mcp_tools
from itertool.config import AppConfig
from itertool.models import ProcessingResult


def setUpModule():
    logger.set_backend("silent")


def tearDownModule():
    logger.set_backend("console")


class _FakeMcp:
    """Collects the functions registered with @mcp.tool(), by name."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(function):
            self.tools[function.__name__] = function
            return function
        return register


class TestTools(unittest.TestCase):
    """Both tools forward their input and the reference context to run_iter."""

    def setUp(self):
        self.mcp = _FakeMcp()
        mcp_tools.register_tools(self.mcp)

    def _call(self, name: str, **arguments):
        with patch.object(mcp_tools, "run_iter", return_value=ProcessingResult(results="RES")) as run_iter, \
                patch.object(mcp_tools, "_build_config", return_value=AppConfig()), \
                patch.object(AppConfig, "apply_to_client"):
            reply = self.mcp.tools[name](**arguments)
        return reply, run_iter.call_args.kwargs

    def test_tool_names(self):
        self.assertEqual(set(self.mcp.tools), {"iter_process_path", "iter_process_text"})

    def test_process_path_forwards_a_folder_and_the_context(self):
        reply, kwargs = self._call(
            "iter_process_path", input_path="docs", node_prompt="p", context_file="glossary.md"
        )
        self.assertEqual(kwargs["input_path"], "docs")
        self.assertEqual(kwargs["context_file"], "glossary.md")
        self.assertEqual(reply, "RES")

    def test_process_text_forwards_the_context(self):
        _, kwargs = self._call("iter_process_text", text="Alpha.", node_prompt="p", context_file="g.md")
        self.assertEqual(kwargs["input_text"], "Alpha.")
        self.assertEqual(kwargs["context_file"], "g.md")

    def test_completion_message_names_the_input(self):
        reply, _ = self._call("iter_process_path", input_path="docs", output_path="out", node_prompt="p")
        self.assertEqual(reply, "Done: docs processed to out.")


if __name__ == "__main__":
    unittest.main()
