"""Output composition for Iter.

Turns the results of a run into output files: the merged layout writes one
file holding every document, the mirror layout one file per document of a
folder. Also names the output of a run when the caller gives none.
"""
from __future__ import annotations

from pathlib import Path

from itertool.models import Preset, ProcessingResult
import itertool.logger as logger

# Introduces each file's part of the results and of the merged output of a folder.
FILE_HEADER = "# File: {name}"


def join_by_file(parts: list[tuple[str, str]], headers: bool) -> str:
    """Join per-file texts, each under its file header when headers is True.

    Args:
        parts: (source file name, text) pairs in corpus order; empty texts are left out.
        headers: Whether to put FILE_HEADER before each text, as for a folder input.

    Returns:
        The joined text; for a single file without headers, that file's text.
    """
    blocks = [
        f"{FILE_HEADER.format(name=name)}\n\n{text}" if headers else text
        for name, text in parts
        if text
    ]
    return "\n\n".join(blocks)


def compose_output(
    preprocessed: str,
    results: str,
    postprocessed: str,
    preset: Preset,
) -> str:
    """Build the output of one document by concatenating its selected sections in order.

    Sections are separated by a Markdown horizontal rule ('---') so the output
    remains valid Markdown even when multiple sections are included.

    Args:
        preprocessed: Phase 1 result.
        results: Phase 2 assembled output.
        postprocessed: Phase 3 result.
        preset: Active preset (reads output_include_* flags).

    Returns:
        Concatenated output string, or empty string if no section is selected.
    """
    parts: list[str] = []
    if preset.output_include_preprocessed and preprocessed:
        parts.append(preprocessed)
    if preset.output_include_processed and results:
        parts.append(results)
    if preset.output_include_postprocessed and postprocessed:
        parts.append(postprocessed)
    return "\n\n---\n\n".join(parts)


def write_outputs(
    result: ProcessingResult,
    corpus_post: str,
    output_path: Path,
    preset: Preset,
    is_folder: bool,
) -> None:
    """Write the outputs of a run in the layout the input and the preset call for.

    A folder input in the 'mirror' output_layout gets one output file per
    document; any other input gets a single merged file.

    Args:
        result: The run's results, one FileResult per document.
        corpus_post: The corpus-wide Phase 3 result, or empty string.
        output_path: The output file, or the output folder of a mirrored folder.
        preset: Active preset (reads output_layout and the output_include_* and save_* flags).
        is_folder: Whether the input is a folder.
    """
    if is_folder and preset.output_layout == "mirror":
        _write_mirror(result, corpus_post, output_path, preset)
    else:
        _write_merged(result, corpus_post, output_path, preset, headers=is_folder)


def _write(path: Path, text: str, label: str) -> None:
    """Write one output file, creating its folder, and log where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.log_save(f"{label} saved to: {path}")


def _side_path(path: Path, suffix: str) -> Path:
    """Path of a side file next to an output file: 'out.md' -> 'out_preprocessed.md'."""
    return path.with_stem(path.stem + suffix)


def _write_merged(result: ProcessingResult, corpus_post: str, out: Path, preset: Preset, headers: bool) -> None:
    """Write one output file holding every document, each under its header when headers is True.

    Each document contributes its own composition (compose_output), so a
    per-file Phase 3 result follows that file's node results. A corpus-wide
    Phase 3 result comes last. A document with nothing selected is left out.

    Args:
        result: The run's results; its aggregate fields fill the side files.
        corpus_post: The corpus-wide Phase 3 result, or empty string.
        out: The output file.
        preset: Active preset (reads the output_include_* and save_* flags).
        headers: Whether to put the file header before each document, as for a folder.
    """
    documents = join_by_file(
        [(f.source, compose_output(f.preprocessed, f.results, f.postprocessed, preset)) for f in result.files],
        headers,
    )
    sections = [documents]
    if preset.output_include_postprocessed:
        sections.append(corpus_post)
    main_output = "\n\n---\n\n".join(s for s in sections if s)
    if main_output:
        _write(out, main_output, "Main output")
    if preset.save_preprocessed_separately and result.preprocessed:
        _write(_side_path(out, "_preprocessed"), result.preprocessed, "Preprocessed output")
    if preset.save_postprocessed_separately and result.postprocessed:
        _write(_side_path(out, "_postprocessed"), result.postprocessed, "Postprocessed output")


def _write_mirror(result: ProcessingResult, corpus_post: str, out_dir: Path, preset: Preset) -> None:
    """Write one output file per document into out_dir, under the document's own name.

    A corpus-wide Phase 3 result belongs to no document, so it is always
    written beside the folder, as '<folder>_postprocessed.md'.

    Args:
        result: The run's results, one FileResult per document.
        corpus_post: The corpus-wide Phase 3 result, or empty string.
        out_dir: The output folder.
        preset: Active preset (reads the output_include_* and save_* flags).
    """
    for file_result in result.files:
        path = out_dir / file_result.source
        main_output = compose_output(
            file_result.preprocessed, file_result.results, file_result.postprocessed, preset
        )
        if main_output:
            _write(path, main_output, "Main output")
        if preset.save_preprocessed_separately and file_result.preprocessed:
            _write(_side_path(path, "_preprocessed"), file_result.preprocessed, "Preprocessed output")
        if preset.save_postprocessed_separately and file_result.postprocessed:
            _write(_side_path(path, "_postprocessed"), file_result.postprocessed, "Postprocessed output")
    if corpus_post:
        _write(out_dir.with_name(f"{out_dir.name}_postprocessed.md"), corpus_post, "Postprocessed output")


def default_output_path(input_path: str, preset: Preset) -> str:
    """Name the output of a run when the caller gives none.

    Args:
        input_path: The input document or folder.
        preset: Active preset (reads output_layout).

    Returns:
        '<name>_processed<ext>' next to a document; for a folder,
        '<folder>_processed' beside it in the mirror layout, or
        '<folder>_processed.md' in the merged layout.
    """
    path = Path(input_path)
    if path.is_dir():
        folder = path.resolve()
        name = f"{folder.name}_processed"
        return str(folder.with_name(name if preset.output_layout == "mirror" else f"{name}.md"))
    return str(path.with_name(f"{path.stem}_processed{path.suffix}"))
