"""The input of a run as a corpus: loading it, parsing it into nodes, guarding its outputs.

A run's input is one Markdown document or a folder of them. Either way it becomes
a Corpus, and its files are parsed one at a time, so no node, section or
pre-pass ever spans two files.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import itertool.logger as logger
from itertool.markdown_utils import parse_into_paragraphs, parse_into_sections
from itertool.models import Corpus, Node, SourceFile

MARKDOWN_SUFFIXES = (".md", ".markdown")

# The UTF-8 byte order mark some editors write at the start of a file.
_BOM = "\N{BYTE ORDER MARK}"


def natural_sort_key(name: str) -> list:
    """Sort key that orders the digits in a name by value: '2.md' before '10.md'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def load_corpus(input_path: str | None = None, input_text: str | None = None) -> Corpus:
    """Load the input of a run.

    Args:
        input_path: A Markdown file, or a folder whose Markdown files
            (.md, .markdown) directly inside it form the corpus. Subfolders
            are not read.
        input_text: Raw Markdown, used instead of input_path when given.

    Returns:
        The corpus: one file for raw text or a file path, one per Markdown
        file for a folder, in natural order of their names. A leading UTF-8
        byte order mark is dropped from every text, so it cannot hide the
        first heading from the parser.

    Raises:
        ValueError: If neither input is usable, or the folder holds no Markdown file.
    """
    if input_text is not None:
        text = input_text.removeprefix(_BOM)
        return Corpus(root=Path.cwd(), files=[SourceFile(name="", text=text)])
    path = Path(input_path) if input_path is not None else None
    if path is not None and path.is_file():
        logger.log_action(f"Loading document from: {input_path}")
        return Corpus(
            root=path.resolve().parent,
            files=[SourceFile(name=path.name, text=path.read_text(encoding="utf-8-sig"))],
        )
    if path is not None and path.is_dir():
        names = sorted(
            (p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in MARKDOWN_SUFFIXES),
            key=natural_sort_key,
        )
        if not names:
            raise ValueError(f"No Markdown file (.md, .markdown) found in the folder: {input_path}")
        logger.log_action(f"Loading {len(names)} documents from: {input_path}")
        return Corpus(
            root=path.resolve(),
            files=[SourceFile(name=n, text=(path / n).read_text(encoding="utf-8-sig")) for n in names],
            is_folder=True,
        )
    raise ValueError(f"Provide a valid input_text or an existing input_path. Got: {input_path}")


def check_output_location(corpus: Corpus, output_path: str | None, layout: str) -> None:
    """Refuse an output that would overwrite the input or join it on the next run.

    Only a folder input is checked. Its output may be neither the input folder
    nor directly inside it: mirrored files would overwrite the originals, and
    any Markdown written there, including the side files written next to the
    output, would be read as part of the corpus by the next run.

    Args:
        corpus: The loaded input.
        output_path: Output folder ('mirror') or file ('merged'), or None.
        layout: The preset's output_layout.

    Raises:
        ValueError: If the output is misplaced, or is a file where a folder
            is needed or the other way round.
    """
    if not corpus.is_folder or not output_path:
        return
    out = Path(output_path).resolve()
    if out == corpus.root or out.parent == corpus.root:
        raise ValueError(
            f"The output {output_path} must not be the input folder or lie directly inside it: "
            f"choose a location outside {corpus.root}."
        )
    if layout == "mirror" and out.is_file():
        raise ValueError(f"The mirror layout writes into a folder, but {output_path} is a file.")
    if layout != "mirror" and out.is_dir():
        raise ValueError(f"The merged layout writes a single file, but {output_path} is a folder.")


def parse_corpus(corpus: Corpus, node_mode: str) -> list[Node]:
    """Parse every file of the corpus into nodes, one file at a time.

    Args:
        corpus: The loaded input.
        node_mode: 'paragraph', 'section', or 'file' (each file is one text node).

    Returns:
        The nodes of all files in order, each tagged with its source file,
        with Node.index 1-based and global across the corpus.
    """
    nodes: list[Node] = []
    for source in corpus.files:
        if node_mode == "file":
            content = source.text.strip()
            parsed = [Node(index=1, node_type="text", content=content)] if content else []
        elif node_mode == "section":
            parsed = parse_into_sections(source.text)
        else:
            parsed = parse_into_paragraphs(source.text)
        offset = len(nodes)
        nodes += [replace(node, index=node.index + offset, source=source.name) for node in parsed]
    return nodes
