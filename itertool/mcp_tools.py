"""MCP tool registrations for Iter."""
from __future__ import annotations

from typing import TYPE_CHECKING

from itertool.checkpoint import CheckpointMismatchError
from itertool.processor import run_iter
from itertool.models import Preset
from itertool.config import AppConfig, resolve_config_path
import itertool.logger as logger

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _build_config(config_file: str | None) -> AppConfig:
    """Load the application configuration and report which file was used.

    An MCP server inherits the working directory of its client, so a config
    resolved implicitly is easy to get wrong and impossible to notice. Logging
    the path turns a silent fallback to field defaults into something visible.

    Args:
        config_file: Explicit path to a config.json, or None to search.

    Returns:
        The loaded AppConfig.
    """
    path = resolve_config_path(config_file)
    if path.exists():
        logger.log_info(f"Using configuration: {path}")
    else:
        logger.log_warning(
            f"No configuration found at {path}; falling back to built-in defaults. "
            f"Pass config_file to select one explicitly."
        )
    return AppConfig.load(path)


def _build_preset(
    preset_file: str | None,
    node_mode: str | None,
    node_prompt: str | None,
    preprocessing_prompt: str | None,
    postprocessing_prompt: str | None,
    compute_similarity: bool | None,
) -> Preset:
    """Build a Preset from an optional preset file plus explicit overrides.

    Mirrors the CLI: a preset supplies the baseline, and any argument the caller
    actually passed overrides it. Arguments left at None change nothing.

    Args:
        preset_file: Path to a preset JSON file, or None for preset defaults.
        node_mode: Node granularity override.
        node_prompt: Node prompt override.
        preprocessing_prompt: Phase 1 prompt override.
        postprocessing_prompt: Phase 3 prompt override.
        compute_similarity: Cosine similarity override.

    Returns:
        The resulting Preset.
    """
    if preset_file:
        preset = Preset.load(preset_file)
        logger.log_info(f"Using preset: {preset_file}")
    else:
        preset = Preset()

    if node_mode is not None:
        preset.node_mode = node_mode
    if node_prompt is not None:
        preset.node_prompt = node_prompt
    if preprocessing_prompt is not None:
        preset.preprocessing_prompt = preprocessing_prompt
    if postprocessing_prompt is not None:
        preset.postprocessing_prompt = postprocessing_prompt
    if compute_similarity is not None:
        preset.compute_similarity = compute_similarity

    if not preset_file:
        # No preset to inherit output composition from: emit the last phase asked for.
        preset.output_include_processed = not preset.postprocessing_prompt
        preset.output_include_postprocessed = bool(preset.postprocessing_prompt)

    return preset


def _mismatch_reply(exc: CheckpointMismatchError) -> str:
    """Explain a progress-file mismatch and how to resolve it on the next call."""
    return (
        f"{exc} Call this tool again with on_mismatch='continue' to keep the saved "
        f"results, or on_mismatch='restart' to discard them."
    )


def register_tools(mcp: FastMCP) -> None:
    """Register all Iter MCP tools with the given FastMCP instance.

    Args:
        mcp: A FastMCP server instance.
    """

    @mcp.tool()
    def iter_process_file(
        input_file: str,
        output_file: str | None = None,
        node_prompt: str | None = None,
        node_mode: str | None = None,
        preprocessing_prompt: str | None = None,
        postprocessing_prompt: str | None = None,
        compute_similarity: bool | None = None,
        config_file: str | None = None,
        preset_file: str | None = None,
        on_mismatch: str | None = None,
    ) -> str:
        """Process a Markdown file node by node using Iter.

        Args:
            input_file: Absolute path to the input Markdown file.
            output_file: Optional path where the main output will be saved.
                If None, the result text is returned directly.
            node_prompt: Prompt applied to each node (paragraph or section).
            node_mode: Node granularity: 'paragraph' or 'section'.
            preprocessing_prompt: Optional prompt for a full-document preprocessing pass.
            postprocessing_prompt: Optional prompt for a full-results postprocessing pass.
            compute_similarity: Whether to compute cosine similarity per node.
            config_file: Optional absolute path to a config.json. Recommended:
                the server inherits the working directory of its client, so an
                implicitly resolved configuration may not be the intended one.
            preset_file: Optional absolute path to a preset JSON file. Supplies
                the baseline settings; the arguments above override it.
            on_mismatch: Only needed after this tool reported a progress file
                from a different run: 'continue' keeps its saved results,
                'restart' discards them.

        Returns:
            Processed text (if output_file is None) or a completion message.
        """
        try:
            config = _build_config(config_file)
            config.apply_to_client()
            preset = _build_preset(
                preset_file,
                node_mode,
                node_prompt,
                preprocessing_prompt,
                postprocessing_prompt,
                compute_similarity,
            )
            result = run_iter(
                input_file=input_file,
                output_file=output_file,
                preset=preset,
                config=config,
                on_mismatch=on_mismatch or "ask",
            )
            if output_file:
                return f"Done: {input_file} processed to {output_file}."
            return result.postprocessed or result.results or result.preprocessed
        except CheckpointMismatchError as exc:
            return _mismatch_reply(exc)
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    def iter_process_text(
        text: str,
        output_file: str | None = None,
        node_prompt: str | None = None,
        node_mode: str | None = None,
        preprocessing_prompt: str | None = None,
        postprocessing_prompt: str | None = None,
        compute_similarity: bool | None = None,
        config_file: str | None = None,
        preset_file: str | None = None,
        on_mismatch: str | None = None,
    ) -> str:
        """Process a Markdown text string node by node using Iter.

        Args:
            text: The Markdown text to process.
            output_file: Optional path to save the processed result.
                If None, the result text is returned directly.
            node_prompt: Prompt applied to each node (paragraph or section).
            node_mode: Node granularity: 'paragraph' or 'section'.
            preprocessing_prompt: Optional prompt for a full-document preprocessing pass.
            postprocessing_prompt: Optional prompt for a full-results postprocessing pass.
            compute_similarity: Whether to compute cosine similarity per node.
            config_file: Optional absolute path to a config.json. Recommended:
                the server inherits the working directory of its client, so an
                implicitly resolved configuration may not be the intended one.
            preset_file: Optional absolute path to a preset JSON file. Supplies
                the baseline settings; the arguments above override it.
            on_mismatch: Only needed after this tool reported a progress file
                from a different run: 'continue' keeps its saved results,
                'restart' discards them.

        Returns:
            Processed text (if output_file is None) or a completion message.
        """
        try:
            config = _build_config(config_file)
            config.apply_to_client()
            preset = _build_preset(
                preset_file,
                node_mode,
                node_prompt,
                preprocessing_prompt,
                postprocessing_prompt,
                compute_similarity,
            )
            result = run_iter(
                input_text=text,
                output_file=output_file,
                preset=preset,
                config=config,
                on_mismatch=on_mismatch or "ask",
            )
            if output_file:
                return f"Processed text saved to {output_file}."
            return result.postprocessed or result.results or result.preprocessed
        except CheckpointMismatchError as exc:
            return _mismatch_reply(exc)
        except Exception as exc:
            return f"Error processing text: {exc}"
