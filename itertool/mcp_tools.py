"""MCP tool registrations for Iter."""
from __future__ import annotations

from itertool.processor import run_iter
from itertool.models import Preset
from itertool.config import AppConfig


def register_tools(mcp) -> None:
    """Register all Iter MCP tools with the given FastMCP instance.

    Args:
        mcp: A FastMCP server instance.
    """

    @mcp.tool()
    def iter_process_file(
        input_file: str,
        output_file: str | None = None,
        node_prompt: str = "",
        node_mode: str = "paragraph",
        preprocessing_prompt: str = "",
        postprocessing_prompt: str = "",
        compute_similarity: bool = False,
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

        Returns:
            Processed text (if output_file is None) or a completion message.
        """
        try:
            preset = Preset(
                node_mode=node_mode,
                node_prompt=node_prompt,
                preprocessing_prompt=preprocessing_prompt,
                postprocessing_prompt=postprocessing_prompt,
                compute_similarity=compute_similarity,
                output_content="postprocessed" if postprocessing_prompt else "processed",
            )
            config = AppConfig.load()
            result = run_iter(
                input_file=input_file,
                output_file=output_file,
                preset=preset,
                config=config,
            )
            if output_file:
                return f"Done: {input_file} processed to {output_file}."
            return result.results or result.postprocessed or result.preprocessed
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    def iter_process_text(
        text: str,
        output_file: str | None = None,
        node_prompt: str = "",
        node_mode: str = "paragraph",
        preprocessing_prompt: str = "",
        postprocessing_prompt: str = "",
        compute_similarity: bool = False,
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

        Returns:
            Processed text (if output_file is None) or a completion message.
        """
        try:
            preset = Preset(
                node_mode=node_mode,
                node_prompt=node_prompt,
                preprocessing_prompt=preprocessing_prompt,
                postprocessing_prompt=postprocessing_prompt,
                compute_similarity=compute_similarity,
                output_content="postprocessed" if postprocessing_prompt else "processed",
            )
            config = AppConfig.load()
            result = run_iter(
                input_text=text,
                output_file=output_file,
                preset=preset,
                config=config,
            )
            if output_file:
                return f"Processed text saved to {output_file}."
            return result.results or result.postprocessed or result.preprocessed
        except Exception as exc:
            return f"Error processing text: {exc}"
