"""Markdown parsing utilities for Iter.

Provides block splitting and classification into typed nodes
(text, heading, code, table, image, rule) for both paragraph and section modes.
"""
from __future__ import annotations

import re
from itertool.models import Node

_FENCE = "```"

# A thematic break: three or more of the same '-', '*' or '_', optionally spaced.
_THEMATIC_BREAK_RE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")

# The underline of a setext heading: only '=' or only '-'.
_SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")

# The start of a list item or a block quote, which a setext underline cannot follow.
_LIST_OR_QUOTE_RE = re.compile(r"^(?:[-*+][ \t]|\d+[.)][ \t]|>)")


# ── Block type predicates ─────────────────────────────────────────────────────

def _is_paragraph_line(line: str) -> bool:
    """Return True if the line can belong to a paragraph, and so to a setext heading.

    Args:
        line: One line of a block.

    Returns:
        False for a blank line, an ATX heading, a table row, a code fence, a
        list item, a block quote or a thematic break; True otherwise.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "|", _FENCE)):
        return False
    return not _LIST_OR_QUOTE_RE.match(stripped) and not _THEMATIC_BREAK_RE.match(stripped)


def _is_setext_heading(block: str) -> bool:
    """Return True if the block is paragraph lines underlined with '=' or '-'.

    Args:
        block: A Markdown text block.

    Returns:
        True for 'Title\\n===' or 'Title\\n---', where every line above the
        underline is a paragraph line.
    """
    lines = block.strip().split("\n")
    return (
        len(lines) >= 2
        and bool(_SETEXT_UNDERLINE_RE.match(lines[-1]))
        and all(_is_paragraph_line(line) for line in lines[:-1])
    )


def is_header(block: str) -> bool:
    """Return True if the block is a Markdown heading.

    Args:
        block: A Markdown text block.

    Returns:
        True if the block starts with one or more '#' characters (ATX), or is
        paragraph lines underlined with '=' or '-' (setext).
    """
    return block.strip().startswith("#") or _is_setext_heading(block)


def heading_text(block: str) -> str:
    """Return the one-line label of a heading block, as used for node IDs.

    Args:
        block: A block for which is_header() is True.

    Returns:
        The first line of an ATX heading, '#' marks included, or the text lines
        of a setext heading joined by a space, without the underline.
    """
    lines = block.strip().split("\n")
    if lines[0].startswith("#"):
        return lines[0].strip()
    return " ".join(line.strip() for line in lines[:-1])


def is_thematic_break(block: str) -> bool:
    """Return True if the block is a Markdown thematic break.

    Args:
        block: A Markdown text block.

    Returns:
        True if the block is a single line of three or more of the same '-',
        '*' or '_', optionally separated by spaces ('---', '***', '- - -').
    """
    return bool(_THEMATIC_BREAK_RE.match(block.strip()))


def has_text(content: str) -> bool:
    """Return True if the content holds anything an AI could process.

    Args:
        content: The text of a node.

    Returns:
        True if it contains at least one letter or digit, in any script.
    """
    return any(ch.isalnum() for ch in content)


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
    return block.strip().startswith(_FENCE)


_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\(([^)]*)\)$")


def is_image_only(block: str) -> bool:
    """Return True if the block contains only a Markdown image.

    A block holding only a link (no leading '!') is ordinary text.

    Args:
        block: A Markdown text block.

    Returns:
        True if the block matches the pattern for a standalone image.
    """
    return bool(_IMAGE_RE.match(block.strip()))


def image_source(block: str) -> str:
    """Return the source of a standalone image block.

    Handles the angle-bracket form for sources with spaces and drops an
    optional quoted title: '![a](<my pic.png> "Title")' gives 'my pic.png'.

    Args:
        block: A block for which is_image_only() is True.

    Returns:
        The image path or URL, as written in the document.

    Raises:
        ValueError: If the block is not a standalone image.
    """
    match = _IMAGE_RE.match(block.strip())
    if not match:
        raise ValueError(f"Not a standalone image block: {block!r}")
    target = match.group(1).strip()
    if target.startswith("<"):
        return target[1:target.index(">")] if ">" in target else target[1:]
    return re.sub(r"""\s+["'].*$""", "", target)


def classify_block(block: str) -> str:
    """Classify a Markdown block into a node type string.

    Args:
        block: A Markdown text block.

    Returns:
        One of: 'heading', 'table', 'code', 'image', 'rule', 'text'.
    """
    if is_header(block):
        return "heading"
    if is_table(block):
        return "table"
    if is_code_block(block):
        return "code"
    if is_image_only(block):
        return "image"
    if is_thematic_break(block):
        return "rule"
    return "text"


# ── Raw block splitting ───────────────────────────────────────────────────────

def _split_setext(block: str) -> list[str]:
    """Split a block after each setext heading it holds.

    A setext underline ends the paragraph lines above it, so 'Para\\n---\\nMore'
    is a heading followed by a paragraph. A code block is returned whole.

    Args:
        block: One block from _split_into_raw_blocks().

    Returns:
        The block's parts, each setext heading ending with its underline.
    """
    if is_code_block(block):
        return [block]
    parts: list[str] = []
    chunk: list[str] = []
    for line in block.split("\n"):
        if chunk and _SETEXT_UNDERLINE_RE.match(line) and all(map(_is_paragraph_line, chunk)):
            parts.append("\n".join([*chunk, line]))
            chunk = []
        else:
            chunk.append(line)
    if chunk:
        parts.append("\n".join(chunk))
    return parts


def _split_into_raw_blocks(text: str) -> list[str]:
    """Split a Markdown document into logical blocks separated by blank lines.

    Preserves fenced code blocks intact (blank lines inside code blocks do not
    trigger a split), and ends a block after each setext heading underline.

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
        if line.strip().startswith(_FENCE):
            in_code = not in_code

        if line.strip() == "" and not in_code:
            if current_block:
                blocks += _split_setext("\n".join(current_block))
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks += _split_setext("\n".join(current_block))

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
            current_heading = heading_text(block)
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
    delimiters (headings, code blocks, tables, images, thematic breaks). Structural delimiters
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
                current_heading = heading_text(block)
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
