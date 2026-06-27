"""Markdown parsing utilities for Iter.

Provides block splitting and classification into typed nodes
(text, heading, code, table, image) for both paragraph and section modes.
"""
from __future__ import annotations

import re
from itertool.models import Node


# ── Block type predicates ─────────────────────────────────────────────────────

def is_header(block: str) -> bool:
    """Return True if the block is a Markdown heading.

    Args:
        block: A Markdown text block.

    Returns:
        True if the block starts with one or more '#' characters.
    """
    return block.strip().startswith("#")


def is_table(block: str) -> bool:
    """Return True if the block is a Markdown table.

    Args:
        block: A Markdown text block.

    Returns:
        True if the block has at least two lines where the first contains '|'
        and the second contains both '|' and '-' (table separator row).
    """
    lines = block.strip().split("\n")
    if len(lines) > 1 and "|" in lines[0] and "|" in lines[1] and "-" in lines[1]:
        return True
    return False


def is_code_block(block: str) -> bool:
    """Return True if the block is a Markdown fenced code block.

    Args:
        block: A Markdown text block.

    Returns:
        True if the block starts with triple backticks.
    """
    return block.strip().startswith("```")


def is_image_or_link_only(block: str) -> bool:
    """Return True if the block contains only a Markdown image or standalone link.

    Args:
        block: A Markdown text block.

    Returns:
        True if the block matches the pattern for a standalone image/link.
    """
    return bool(re.match(r"^!?\[.*?\]\(.*?\)$", block.strip()))


def classify_block(block: str) -> str:
    """Classify a Markdown block into a node type string.

    Args:
        block: A Markdown text block.

    Returns:
        One of: 'heading', 'table', 'code', 'image', 'text'.
    """
    if is_header(block):
        return "heading"
    if is_table(block):
        return "table"
    if is_code_block(block):
        return "code"
    if is_image_or_link_only(block):
        return "image"
    return "text"


# ── Raw block splitting ───────────────────────────────────────────────────────

def _split_into_raw_blocks(text: str) -> list[str]:
    """Split a Markdown document into logical blocks separated by blank lines.

    Preserves fenced code blocks intact (blank lines inside code blocks do not
    trigger a split).

    Args:
        text: Full Markdown document content.

    Returns:
        List of non-empty text blocks.
    """
    lines = text.split("\n")
    blocks: list[str] = []
    current_block: list[str] = []
    in_code = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code

        if line.strip() == "" and not in_code:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


# ── Paragraph mode parsing ────────────────────────────────────────────────────

def parse_into_paragraphs(text: str) -> list[Node]:
    """Parse a Markdown document into paragraph-level Node objects.

    Each logical block separated by blank lines becomes a distinct Node.
    Node index is 1-based. The heading_prefix of each node is set to the
    most recently seen heading, enabling meaningful node IDs.

    Args:
        text: Full Markdown document content.

    Returns:
        List of Node objects, one per logical block.
    """
    raw_blocks = _split_into_raw_blocks(text)
    nodes: list[Node] = []
    current_heading = ""
    counter = 1

    for block in raw_blocks:
        block_type = classify_block(block)
        if block_type == "heading":
            current_heading = block.strip()
        node = Node(
            index=counter,
            node_type=block_type,
            content=block,
            heading_prefix=current_heading,
        )
        nodes.append(node)
        counter += 1

    return nodes


# ── Section mode parsing ──────────────────────────────────────────────────────

def parse_into_sections(text: str) -> list[Node]:
    """Parse a Markdown document into section-level Node objects.

    A section is the contiguous run of text paragraphs between structural
    delimiters (headings, code blocks, tables, images). Structural delimiters
    themselves are yielded as individual nodes of their respective types.

    The index is assigned globally (1-based across all nodes, including
    structural ones). The heading_prefix of each text-section node is set to
    the most recently seen heading.

    Args:
        text: Full Markdown document content.

    Returns:
        List of Node objects; text sections have node_type='text',
        structural elements retain their own type.
    """
    raw_blocks = _split_into_raw_blocks(text)
    nodes: list[Node] = []
    buffer: list[str] = []
    current_heading = ""
    counter = 1

    def _flush_buffer() -> None:
        nonlocal counter
        if buffer:
            combined = "\n\n".join(buffer)
            nodes.append(Node(
                index=counter,
                node_type="text",
                content=combined,
                heading_prefix=current_heading,
            ))
            counter += 1
            buffer.clear()

    for block in raw_blocks:
        block_type = classify_block(block)

        if block_type == "text":
            buffer.append(block)
        else:
            # Flush accumulated text paragraphs as one section node
            _flush_buffer()
            # Track latest heading for subsequent text sections
            if block_type == "heading":
                current_heading = block.strip()
            # Emit the structural block as its own node
            nodes.append(Node(
                index=counter,
                node_type=block_type,
                content=block,
                heading_prefix=current_heading,
            ))
            counter += 1

    # Flush any remaining text after the last structural block
    _flush_buffer()

    return nodes


# ── AI response cleanup ───────────────────────────────────────────────────────

def clean_ai_response(text: str) -> str:
    """Strip residual Markdown code fence wrappers from an AI response.

    Some models wrap their output in triple backticks even when not asked to.
    This function removes those wrappers if present.

    Args:
        text: Raw AI response text.

    Returns:
        Cleaned text with code fence wrappers removed.
    """
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[11:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
