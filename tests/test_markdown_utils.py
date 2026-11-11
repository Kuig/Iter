"""Tests for block splitting, classification and AI response cleaning."""
from __future__ import annotations

import unittest

from itertool.markdown_utils import (
    _split_into_raw_blocks,
    classify_block,
    clean_ai_response,
    has_text,
    heading_text,
    image_source,
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
    """Every block gets exactly one of the six tags."""

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

    def test_link_alone_is_text(self):
        self.assertEqual(classify_block("[docs](https://example.com)"), "text")

    def test_two_images_in_one_block_are_text(self):
        self.assertEqual(classify_block("![a](a.png) and ![b](b.png)"), "text")


class TestThematicBreak(unittest.TestCase):
    """A thematic break is a rule node, never text the AI is asked to process."""

    def test_rules(self):
        for block in ("---", "***", "___", "- - -", "*****", " _ _ _ "):
            with self.subTest(block=block):
                self.assertEqual(classify_block(block), "rule")

    def test_too_short_or_mixed_is_text(self):
        for block in ("--", "-*-", "- item", "*emphasis*"):
            with self.subTest(block=block):
                self.assertEqual(classify_block(block), "text")

    def test_section_mode_splits_at_a_rule(self):
        nodes = parse_into_sections("One.\n\n---\n\nTwo.")
        self.assertEqual([n.node_type for n in nodes], ["text", "rule", "text"])

    def test_rule_between_code_and_heading_is_not_text(self):
        doc = FENCE + "bash\nls\n" + FENCE + "\n\n---\n\n## Next"
        nodes = parse_into_sections(doc)
        self.assertEqual([n.node_type for n in nodes], ["code", "rule", "heading"])

    def test_rule_after_a_table_is_not_text(self):
        nodes = parse_into_sections("| a |\n|---|\n| 1 |\n\n---\n\n## Next")
        self.assertEqual([n.node_type for n in nodes], ["table", "rule", "heading"])


class TestSetextHeading(unittest.TestCase):
    """A line underlined with '=' or '-' is a heading, as CommonMark reads it."""

    def test_underlined_with_equals_or_dashes(self):
        self.assertEqual(classify_block("Title\n==="), "heading")
        self.assertEqual(classify_block("Title\n---"), "heading")

    def test_multi_line_heading(self):
        self.assertEqual(classify_block("Long\ntitle\n="), "heading")

    def test_underline_ends_the_block(self):
        blocks = _split_into_raw_blocks("Para\n---\nMore text.")
        self.assertEqual(blocks, ["Para\n---", "More text."])
        self.assertEqual([classify_block(b) for b in blocks], ["heading", "text"])

    def test_list_followed_by_dashes_is_not_a_heading(self):
        self.assertEqual(classify_block("- item\n---"), "text")

    def test_code_block_is_not_split(self):
        code = FENCE + "\nTitle\n---\n" + FENCE
        self.assertEqual(_split_into_raw_blocks(code), [code])

    def test_heading_text_is_one_line(self):
        self.assertEqual(heading_text("Long\ntitle\n==="), "Long title")
        self.assertEqual(heading_text("## Title"), "## Title")

    def test_following_node_id_names_the_heading(self):
        nodes = parse_into_sections("Title\n=====\n\nBody.")
        self.assertEqual(nodes[1].node_id, "2 - Title")


class TestHasText(unittest.TestCase):
    """Only content holding a letter or a digit is worth an AI call."""

    def test_letters_and_digits_in_any_script(self):
        for content in ("Text", "42", "è", "日本"):
            with self.subTest(content=content):
                self.assertTrue(has_text(content))

    def test_punctuation_and_markup_only(self):
        for content in ("", "---", "===", "* * *", "#", "> ", "..."):
            with self.subTest(content=content):
                self.assertFalse(has_text(content))


class TestImageSource(unittest.TestCase):
    """The source is extracted the way Markdown writes it."""

    def test_plain(self):
        self.assertEqual(image_source("![alt](pics/a.png)"), "pics/a.png")

    def test_title_is_dropped(self):
        self.assertEqual(image_source('![alt](a.png "A title")'), "a.png")

    def test_angle_brackets_allow_spaces(self):
        self.assertEqual(image_source("![alt](<my pics/a b.png> 'T')"), "my pics/a b.png")

    def test_url(self):
        self.assertEqual(image_source("![](https://example.com/a.png)"), "https://example.com/a.png")

    def test_not_an_image(self):
        with self.assertRaises(ValueError):
            image_source("[docs](a.md)")


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
