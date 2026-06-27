"""Data models for Iter: Preset, ProcessingResult, Node, and similarity helpers."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ── Preset ─────────────────────────────────────────────────────────────────────

@dataclass
class Preset:
    """A saved configuration preset for the Iter GUI.

    Stores all three prompt texts, node mode, processing options, and
    output composition settings. Model settings (provider, model name, etc.)
    are stored separately in config.json and are NOT part of a preset,
    except for the system_prompt which can be customized per-task.

    Attributes:
        name: Human-readable preset name.
        system_prompt: System instruction for all AI calls in this preset.
        node_mode: Granularity of processing nodes ('paragraph' or 'section').
        preprocessing_prompt: Prompt for Phase 1 (whole-document AI call).
        node_prompt: Prompt applied to each node in Phase 2.
        postprocessing_prompt: Prompt for Phase 3 (whole-results AI call).
        include_preprocessed_context: If True, inject the Phase 1 result as
            a prior assistant message when calling the node prompt.
        heading_handling: How to handle heading blocks in Phase 2.
            'discard': skip headings entirely.
            'report_unprocessed': pass headings through unchanged.
            'standalone_nodes': treat each heading as its own node.
            'merge_next': prepend heading text to the next text node.
        non_text_handling: How to handle tables/code/images in Phase 2.
            'discard': skip non-text blocks entirely.
            'report_unprocessed': pass them through unchanged.
            'standalone_nodes': treat each as its own node.
            'part_of_section': include in the surrounding section text (section mode only).
        compute_similarity: Whether to compute cosine similarity per node.
        show_similarity_separator: Whether to show the §N separator with similarity score.
        postprocess_attach_input: Attach the original input doc to Phase 3 prompt.
        postprocess_attach_preprocessed: Attach Phase 1 result to Phase 3 prompt.
        postprocess_attach_results: Attach Phase 2 results to Phase 3 prompt.
        output_include_preprocessed: Include Phase 1 result in the main output file.
        output_include_processed: Include Phase 2 node results in the main output file.
        output_include_postprocessed: Include Phase 3 result in the main output file.
            Selected sections are appended in order, separated by '---'.
        save_preprocessed_separately: Save Phase 1 result to a separate file.
        save_postprocessed_separately: Save Phase 3 result to a separate file.
        interlace_input_output: Interlace the original input node content with the processed output content.
    """

    name: str = ""
    system_prompt: str = ""
    node_mode: str = "paragraph"
    preprocessing_prompt: str = ""
    node_prompt: str = ""
    postprocessing_prompt: str = ""
    include_preprocessed_context: bool = False
    heading_handling: str = "report_unprocessed"
    non_text_handling: str = "report_unprocessed"
    compute_similarity: bool = False
    show_similarity_separator: bool = True
    postprocess_attach_input: bool = False
    postprocess_attach_preprocessed: bool = False
    postprocess_attach_results: bool = True
    output_include_preprocessed: bool = False
    output_include_processed: bool = True
    output_include_postprocessed: bool = False
    save_preprocessed_separately: bool = False
    save_postprocessed_separately: bool = False
    interlace_input_output: bool = False

    @classmethod
    def load(cls, path: str | Path) -> Preset:
        """Load a preset from a JSON file.

        Args:
            path: Path to the preset JSON file.

        Returns:
            A Preset instance populated from the file.

        Raises:
            FileNotFoundError: If the preset file does not exist.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        """Save this preset to a JSON file.

        Args:
            path: Path where the preset JSON file will be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


# ── Node ───────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    """A single processing unit extracted from the document.

    Attributes:
        index: 1-based sequential node index.
        node_type: Block type: 'text', 'heading', 'code', 'table', 'image'.
        content: Raw text content of the node.
        heading_prefix: Nearest heading text above this node, used for node IDs.
    """

    index: int
    node_type: str
    content: str
    heading_prefix: str = ""

    @property
    def node_id(self) -> str:
        """Build a human-readable node identifier.

        Returns:
            A string like '3 - Introduction' or just '3' if no heading is available.
        """
        if self.heading_prefix:
            clean = self.heading_prefix.lstrip("#").strip()
            return f"{self.index} - {clean}"
        return str(self.index)


# ── ProcessingResult ──────────────────────────────────────────────────────────

@dataclass
class NodeResult:
    """Result for a single processed node.

    Attributes:
        node: The original node that was processed.
        output: The AI-generated response text, or the original text if skipped.
        similarity: Cosine similarity score between node and output (None if not computed).
        was_processed: Whether the node was sent to the AI (False means it was passed through).
        input_chars: Number of input characters sent to the AI (0 if skipped).
        output_chars: Number of output characters received from the AI (0 if skipped).
    """

    node: Node
    output: str
    similarity: float | None = None
    was_processed: bool = True
    input_chars: int = 0
    output_chars: int = 0


@dataclass
class ProcessingResult:
    """Holds all intermediate and final results from a full Iter run.

    Attributes:
        preprocessed: Output from Phase 1 (preprocessing prompt), or empty string.
        node_results: List of per-node results from Phase 2.
        results: Assembled Phase 2 output string (nodes joined with separators).
        postprocessed: Output from Phase 3 (postprocessing prompt), or empty string.
    """

    preprocessed: str = ""
    node_results: list[NodeResult] = field(default_factory=list)
    results: str = ""
    postprocessed: str = ""


# ── Cosine Similarity ─────────────────────────────────────────────────────────

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.

    Returns:
        Cosine similarity score in [-1.0, 1.0]. Returns 0.0 for empty vectors.
    """
    if not vec1 or not vec2:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_similarity_indicator(score: float) -> str:
    """Return a colored circle emoji based on the similarity score.

    Args:
        score: Cosine similarity score.

    Returns:
        '🟢' (>=0.85), '🟡' (>=0.70), or '🔴' (below 0.70).
    """
    if score >= 0.85:
        return "🟢"
    if score >= 0.70:
        return "🟡"
    return "🔴"
