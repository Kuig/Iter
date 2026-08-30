"""Tests for block splitting, classification and AI response cleaning."""
from __future__ import annotations

import unittest

from itertool.markdown_utils import (
    _split_into_raw_blocks,
    classify_block,
    clean_ai_response,
    parse_into_paragraphs,
    parse_into_sections,
)

FENCE = "```"


class TestSplitIntoRawBlocks(unittest.TestCase):
    """Blank lines separate blocks, except inside a fenced code block."""

    def test_splits_on_blank_lines(self):
        blocks = _split_into_raw_blocks("One.\n\nTwo.\n\nThree.")
        self.assertEqual(len(blocks), 3)

    def test_collapses_repeated_blank_lines(self):
        blocks = _split_into_raw_blocks("One.\n\n\n\nTwo.")
        self.assertEqual(len(blocks), 2)

    def test_blank_line_inside_fence_does_not_split(self):
        text = "Before.\n\n" + FENCE + "python\na = 1\n\nb = 2\n" + FENCE + "\n\nAfter."
        blocks = _split_into_raw_blocks(text)
        self.assertEqual(len(blocks), 3)
        self.assertIn("b = 2", blocks[1])
        self.assertEqual(classify_block(blocks[1]), "code")

    def test_ignores_trailing_whitespace_only_content(self):
        self.assertEqual(_split_into_raw_blocks("   \n\n  \n"), [])


class TestClassifyBlock(unittest.TestCase):
    """Every block gets exactly one of the five tags."""

    def test_heading(self):
        self.assertEqual(classify_block("## Title"), "heading")

    def test_table(self):
        self.assertEqual(classify_block("| a | b |\n| - | - |\n| 1 | 2 |"), "table")

    def test_code(self):
        self.assertEqual(classify_block(FENCE + "python\na = 1\n" + FENCE), "code")

    def test_image(self):
        self.assertEqual(classify_block("![alt](img.png)"), "image")

    def test_text(self):
        self.assertEqual(classify_block("Just a paragraph."), "text")


class TestCleanAiResponse(unittest.TestCase):
    """Models wrap answers in a markdown fence unprompted; it must come off."""

    def test_strips_markdown_wrapper(self):
        wrapped = FENCE + "markdown\nHello world.\n" + FENCE
        self.assertEqual(clean_ai_response(wrapped), "Hello world.")

    def test_leaves_unwrapped_text_alone(self):
        self.assertEqual(clean_ai_response("Hello world."), "Hello world.")

    def test_preserves_a_genuine_code_block(self):
        code = FENCE + "python\na = 1\n" + FENCE
        self.assertIn("a = 1", clean_ai_response(code))


class TestNodeIndex(unittest.TestCase):
    """Node.index is 1-based and global across both node modes."""

    DOC = "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n\n| a |\n| - |\n"

    def test_paragraph_mode_indices(self):
        nodes = parse_into_paragraphs(self.DOC)
        self.assertEqual([n.index for n in nodes], list(range(1, len(nodes) + 1)))
        self.assertEqual(nodes[0].index, 1)

    def test_section_mode_indices(self):
        nodes = parse_into_sections(self.DOC)
        self.assertEqual([n.index for n in nodes], list(range(1, len(nodes) + 1)))

    def test_section_mode_buffers_consecutive_text(self):
        paragraphs = parse_into_paragraphs(self.DOC)
        sections = parse_into_sections(self.DOC)
        text_paragraphs = [n for n in paragraphs if n.node_type == "text"]
        text_sections = [n for n in sections if n.node_type == "text"]
        self.assertEqual(len(text_paragraphs), 2)
        self.assertEqual(len(text_sections), 1)
        self.assertIn("First paragraph.", text_sections[0].content)
        self.assertIn("Second paragraph.", text_sections[0].content)


if __name__ == "__main__":
    unittest.main()
