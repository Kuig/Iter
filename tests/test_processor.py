"""Tests for the node-processing pipeline.

Every AI call is stubbed: no test in this suite may reach call_ai.
"""
from __future__ import annotations

import unittest

import itertool.logger as logger
import itertool.processor as processor
from itertool.config import AppConfig
from itertool.models import Node, Preset

FENCE = "```"
EM_DASH = "\u2014"  # what the separator used before 0.1.2, and must never contain again


def setUpModule():
    """Run the pipeline quietly: this is what the silent backend is for."""
    logger.set_backend("silent")


def tearDownModule():
    logger.set_backend("console")


class _StubAi:
    """Replaces _call_ai for the duration of a test and records its prompts."""

    def __init__(self):
        self.prompts: list[str] = []
        self._original = None

    def __enter__(self):
        self._original = processor._call_ai

        def fake(prompt, config, preset, messages=None):
            self.prompts.append(prompt)
            return "[AI]"

        processor._call_ai = fake
        return self

    def __exit__(self, *exc):
        processor._call_ai = self._original


class _CaptureWarnings:
    """Collects logger.log_warning calls made through the processor module."""

    def __init__(self):
        self.messages: list[str] = []
        self._original = None

    def __enter__(self):
        self._original = processor.logger.log_warning
        processor.logger.log_warning = self.messages.append
        return self

    def __exit__(self, *exc):
        processor.logger.log_warning = self._original


class TestBuildSeparator(unittest.TestCase):
    """The separator is part of the output contract, so its shape is pinned."""

    NODE = Node(index=3, node_type="text", content="x")

    def test_hidden_when_not_requested(self):
        self.assertEqual(processor._build_separator(self.NODE, 0.9, False), "")

    def test_without_similarity(self):
        self.assertEqual(processor._build_separator(self.NODE, None, True), "# §3")

    def test_with_similarity_uses_a_plain_hyphen(self):
        out = processor._build_separator(self.NODE, 0.842, True)
        self.assertTrue(out.startswith("# §3 - Cosine similarity: 0.842"))
        self.assertNotIn(EM_DASH, out)


class TestMergeNextPrePass(unittest.TestCase):
    """The pre-pass merges what it can; the rest reaches _process_node."""

    def _run(self, doc: str):
        preset = Preset(node_prompt="p", node_mode="paragraph", heading_handling="merge_next")
        with _StubAi():
            results, _ = processor._run_node_processing(doc, preset, AppConfig(), "")
        return results

    def test_heading_followed_by_text_is_merged(self):
        results = self._run("# Title\n\nSome text.\n")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].node.node_type, "text")
        self.assertIn("# Title", results[0].node.content)
        self.assertIn("Some text.", results[0].node.content)

    def test_trailing_heading_reaches_the_residual_branch(self):
        results = self._run("Some text.\n\n# Trailing\n")
        heading = [r for r in results if r.node.node_type == "heading"]
        self.assertEqual(len(heading), 1)
        self.assertFalse(heading[0].was_processed)
        self.assertEqual(heading[0].output, "# Trailing")

    def test_heading_followed_by_a_table_reaches_the_residual_branch(self):
        results = self._run("# Title\n\n| a |\n| - |\n")
        heading = [r for r in results if r.node.node_type == "heading"]
        self.assertEqual(len(heading), 1)
        self.assertFalse(heading[0].was_processed)


class TestPartOfSection(unittest.TestCase):
    """part_of_section needs section mode, and says so when it does not have it."""

    DOC_LEADING_TABLE = "| a |\n| - |\n\nSome text.\n"
    DOC_TRAILING_TABLE = "Some text.\n\n| a |\n| - |\n"

    def _run(self, doc: str, node_mode: str):
        preset = Preset(
            node_prompt="p",
            node_mode=node_mode,
            non_text_handling="part_of_section",
        )
        with _CaptureWarnings() as warnings, _StubAi():
            results, _ = processor._run_node_processing(doc, preset, AppConfig(), "")
        return results, warnings.messages

    def test_paragraph_mode_warns(self):
        _, warnings = self._run(self.DOC_TRAILING_TABLE, "paragraph")
        self.assertEqual(len(warnings), 1)
        self.assertIn("part_of_section", warnings[0])
        self.assertIn("section mode", warnings[0])

    def test_section_mode_does_not_warn(self):
        _, warnings = self._run(self.DOC_TRAILING_TABLE, "section")
        self.assertEqual(warnings, [])

    def test_section_mode_absorbs_a_following_non_text_block(self):
        results, _ = self._run(self.DOC_TRAILING_TABLE, "section")
        self.assertEqual(len(results), 1)
        self.assertIn("| a |", results[0].node.content)

    def test_leading_non_text_block_reaches_the_residual_branch(self):
        results, _ = self._run(self.DOC_LEADING_TABLE, "section")
        table = [r for r in results if r.node.node_type == "table"]
        self.assertEqual(len(table), 1)
        self.assertFalse(table[0].was_processed)

    def test_paragraph_mode_passes_non_text_through_unchanged(self):
        results, _ = self._run(self.DOC_TRAILING_TABLE, "paragraph")
        table = [r for r in results if r.node.node_type == "table"]
        self.assertEqual(len(table), 1)
        self.assertFalse(table[0].was_processed)
        self.assertIn("| a |", table[0].output)


class TestOtherHandlingRules(unittest.TestCase):
    """discard, report_unprocessed and standalone_nodes stay in _process_node."""

    DOC = "# Title\n\nSome text.\n"

    def _run(self, heading_handling: str):
        preset = Preset(node_prompt="p", node_mode="paragraph", heading_handling=heading_handling)
        with _StubAi():
            results, _ = processor._run_node_processing(self.DOC, preset, AppConfig(), "")
        return results

    def test_discard_drops_the_heading_output(self):
        results = self._run("discard")
        heading = [r for r in results if r.node.node_type == "heading"][0]
        self.assertEqual(heading.output, "")
        self.assertFalse(heading.was_processed)

    def test_report_unprocessed_keeps_it_verbatim(self):
        results = self._run("report_unprocessed")
        heading = [r for r in results if r.node.node_type == "heading"][0]
        self.assertEqual(heading.output, "# Title")
        self.assertFalse(heading.was_processed)

    def test_standalone_nodes_sends_it_to_the_ai(self):
        results = self._run("standalone_nodes")
        heading = [r for r in results if r.node.node_type == "heading"][0]
        self.assertTrue(heading.was_processed)
        self.assertEqual(heading.output, "[AI]")


class TestPhaseSkipping(unittest.TestCase):
    """A phase is skipped purely by its prompt being empty."""

    def test_empty_node_prompt_skips_phase_two(self):
        preset = Preset(node_prompt="   ")
        with _StubAi() as ai:
            results, text = processor._run_node_processing("Some text.", preset, AppConfig(), "")
        self.assertEqual(results, [])
        self.assertEqual(text, "")
        self.assertEqual(ai.prompts, [])


if __name__ == "__main__":
    unittest.main()
