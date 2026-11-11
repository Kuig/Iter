"""Core processing pipeline for Iter.

Implements the three-phase pipeline over a corpus (one document, or the
Markdown files of a folder):
  Phase 1 - Preprocessing: single AI call on each full document.
  Phase 2 - Node Processing: same prompt applied independently to each node.
  Phase 3 - Postprocessing: single AI call on the assembled results, per
    document or for the whole corpus.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from unified_ai_client import call_ai
from unified_ai_client import get_embedding as _get_embedding_vec

from itertool.checkpoint import (
    Checkpoint,
    file_digest,
    fingerprint,
    node_keys,
    progress_path,
    text_key,
)
from itertool.config import AppConfig
from itertool.corpus import check_output_location, load_corpus, parse_corpus
from itertool.images import resolve_images
from itertool.models import (
    Corpus,
    FileResult,
    Node,
    NodeResult,
    Preset,
    ProcessingResult,
    cosine_similarity,
    get_similarity_indicator,
)
from itertool.markdown_utils import clean_ai_response
from itertool.output import join_by_file, write_outputs
import itertool.logger as logger

# Introduces the reference context at the end of the system prompt.
_CONTEXT_HEADER = "=== REFERENCE CONTEXT ==="


# ── AI helpers ────────────────────────────────────────────────────────────────

@dataclass
class _AiReply:
    """A cleaned AI response with the cost of obtaining it.

    Attributes:
        text: Cleaned response text.
        elapsed: Wall-clock seconds of the call, retries included.
        input_tokens: Prompt tokens reported by the provider (0 if unknown).
        output_tokens: Response tokens reported by the provider (0 if unknown).
    """

    text: str
    elapsed: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _RunState:
    """Mutable state shared by the nodes of one Phase 2 run.

    Attributes:
        similarity_enabled: Whether cosine similarity is still computed. Turned
            off after the first embedding failure, so a broken embedding server
            costs one retry cycle per run instead of two per node.
        checkpoint: Progress file to restore from and record into, or None.
    """

    similarity_enabled: bool
    checkpoint: Checkpoint | None = None


def _format_duration(seconds: float) -> str:
    """Format a duration as '12.3 s' below one minute, '4 min 05 s' above."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes} min {secs:02d} s"


def _reply_stats(input_chars: int, reply: _AiReply) -> str:
    """Summarise one AI call for the log: sizes, duration and, if known, tokens."""
    stats = f"in: {input_chars}, out: {len(reply.text)} chars, {_format_duration(reply.elapsed)}"
    if reply.input_tokens or reply.output_tokens:
        stats += f", tokens in/out: {reply.input_tokens}/{reply.output_tokens}"
    return stats


def _call_ai(
    prompt: str,
    config: AppConfig,
    preset: Preset,
    *,
    messages: list[dict] | None = None,
    file_path: Path | None = None,
) -> _AiReply:
    """Make a single AI call and return the cleaned response with its timing.

    Args:
        prompt: User prompt text.
        config: Application configuration (provider, model, temperature, thinking).
        preset: Active preset (provides system_prompt).
        messages: Optional prior conversation history for context injection.
        file_path: Optional file attached to the prompt (an image node's file).

    Returns:
        The cleaned response, its duration and the reported token counts.
    """
    system_prompt = preset.system_prompt or None
    start = time.perf_counter()
    response = call_ai(
        provider=config.provider,
        model=config.model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
        file_path=str(file_path) if file_path else None,
        temperature=config.temperature,
        thinking=config.thinking,
    )
    return _AiReply(
        text=clean_ai_response(response.text),
        elapsed=time.perf_counter() - start,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def _with_reference_context(preset: Preset, context: str) -> Preset:
    """Return the preset with the reference context appended to its system prompt.

    Every AI call of the run takes its system prompt from the preset, so this
    puts the context in front of the model in all three phases.

    Args:
        preset: Active preset.
        context: Text of the reference context file; blank means none.

    Returns:
        The preset itself when there is no context, otherwise a copy whose
        system_prompt ends with the context block.
    """
    if not context.strip():
        return preset
    block = f"{_CONTEXT_HEADER}\n{context}"
    system_prompt = f"{preset.system_prompt}\n\n{block}" if preset.system_prompt else block
    return replace(preset, system_prompt=system_prompt)


def _get_embedding(text: str, config: AppConfig, file_path: Path | None = None) -> list[float]:
    """Compute an embedding vector using the configured embedding model.

    Args:
        text: Text to embed.
        config: Application configuration (reads embedding_provider and embedding_model).
        file_path: Optional file embedded together with the text into one
            vector; needs a multimodal embedding model.

    Returns:
        Embedding vector as a list of floats.

    Raises:
        Exception: Whatever UnifiedAiClient raises once its retries are spent.
    """
    return _get_embedding_vec(
        provider=config.embedding_provider,
        model=config.embedding_model,
        text=text,
        file_path=str(file_path) if file_path else None,
    )


def _compute_similarity(
    original: str,
    output: str,
    config: AppConfig,
    state: _RunState,
    file_path: Path | None = None,
) -> float | None:
    """Return the cosine similarity of two texts, or None if it is not computed.

    The first embedding failure is logged once and disables similarity for the
    rest of the run.

    Args:
        original: Node content sent to the AI.
        output: AI response for that node.
        config: Application configuration (embedding provider and model).
        state: Run state; similarity_enabled is read and may be switched off.
        file_path: File sent to the AI with the node, embedded with original.

    Returns:
        The similarity score, or None when disabled or failed.
    """
    if not state.similarity_enabled:
        return None
    try:
        vec1 = _get_embedding(original, config, file_path)
        vec2 = _get_embedding(output, config)
    except Exception as exc:
        state.similarity_enabled = False
        logger.log_warning(
            f"Embedding request failed ({config.embedding_model}): {exc}. "
            f"Cosine similarity is disabled for the rest of this run."
        )
        return None
    return cosine_similarity(vec1, vec2)


# ── Phase 1: Preprocessing ────────────────────────────────────────────────────

def _preprocess(
    document_text: str,
    preset: Preset,
    config: AppConfig,
    checkpoint: Checkpoint | None = None,
) -> str:
    """Make the Phase 1 call for one document, or restore it from the progress file.

    Args:
        document_text: Full text of the document.
        preset: Active preset.
        config: Application configuration.
        checkpoint: Progress file to restore from and record into, or None.

    Returns:
        AI response string.
    """
    key = text_key(document_text)
    if checkpoint and key in checkpoint.preprocessed:
        logger.log_action("Preprocessing restored from the progress file.")
        return checkpoint.preprocessed[key]

    prompt = f"{preset.preprocessing_prompt}\n\n{document_text}"
    reply = _call_ai(prompt, config, preset)
    logger.log_success(f"Preprocessing complete ({_reply_stats(len(document_text), reply)}).")
    if checkpoint:
        checkpoint.record_preprocessed(key, reply.text, reply.elapsed)
    return reply.text


def _run_preprocessing(
    corpus: Corpus,
    preset: Preset,
    config: AppConfig,
    checkpoint: Checkpoint | None = None,
) -> dict[str, str]:
    """Run Phase 1: a single AI call over each full document of the corpus.

    Skipped if preset.preprocessing_prompt is empty.

    Args:
        corpus: The input documents.
        preset: Active preset.
        config: Application configuration.
        checkpoint: Progress file to restore from and record into, or None.

    Returns:
        AI response string by source file name, or an empty dict if skipped.
    """
    if not preset.preprocessing_prompt.strip():
        return {}

    logger.log_separator()
    logger.log_info("Phase 1: Preprocessing")
    preprocessed: dict[str, str] = {}
    for source in corpus.files:
        if corpus.is_folder:
            logger.log_action(f"File: {source.name}")
        preprocessed[source.name] = _preprocess(source.text, preset, config, checkpoint)
    return preprocessed


# ── Phase 2: Node Processing ──────────────────────────────────────────────────

def _build_separator(node: Node, similarity: float | None, show: bool) -> str:
    """Build the optional §N separator line for a node result.

    Args:
        node: The node being processed.
        similarity: Cosine similarity score, or None if not computed.
        show: Whether the separator should be shown at all.

    Returns:
        A formatted separator string, or empty string if show is False.
    """
    if not show:
        return ""
    parts = [f"# §{node.node_id}"]
    if similarity is not None:
        indicator = get_similarity_indicator(similarity)
        parts.append(f"Cosine similarity: {similarity:.3f} {indicator}")
    return " - ".join(parts)


def _handling_for(node: Node, preset: Preset) -> str:
    """Return the handling rule that governs a structural (non-text) node.

    Args:
        node: A node whose node_type is not 'text'.
        preset: Active preset (reads the three handling fields).

    Returns:
        The handling value of heading_handling, image_handling or non_text_handling.
    """
    if node.node_type == "heading":
        return preset.heading_handling
    if node.node_type == "image":
        return preset.image_handling
    return preset.non_text_handling


def _process_node(
    node: Node,
    preset: Preset,
    config: AppConfig,
    preprocessed_context: str,
    state: _RunState,
    key: str = "",
    image_path: Path | None = None,
) -> NodeResult:
    """Process a single node through the AI with optional similarity computation.

    Applies heading/non-text handling rules before deciding whether to call AI,
    and restores the node from the progress file when it was done before.

    Args:
        node: The node to process.
        preset: Active preset (reads handling rules and options).
        config: Application configuration.
        preprocessed_context: Phase 1 result, used if include_preprocessed_context is True.
        state: Run state shared across nodes (similarity, progress file).
        key: The node's progress-file key, from checkpoint.node_keys().
        image_path: File attached to the AI call, for an image node.

    Returns:
        A NodeResult with the output and optional similarity score.
    """
    is_structural = node.node_type != "text"

    # --- Determine handling for structural (non-text) nodes ---
    if is_structural:
        handling = _handling_for(node, preset)

        if handling == "discard":
            logger.log_action("  Discarded")
            return NodeResult(node=node, output="", was_processed=False)

        if handling == "report_unprocessed":
            logger.log_action("  Passed through unchanged")
            return NodeResult(node=node, output=node.content, was_processed=False)

        if handling == "standalone_nodes":
            logger.log_action("  Processing as standalone")
            # Falls through to AI call below

        if handling == "merge_next":
            # The pre-pass in _run_node_processing merged every heading that had
            # a following text node. What arrives here is the residue: a heading
            # at the end of the document, or one followed by another heading, a
            # table or a code block. There is nothing to merge with.
            logger.log_action("  Nothing to merge with, passed through unchanged")
            return NodeResult(node=node, output=node.content, was_processed=False)

        # The pre-pass absorbed every non-text block that followed a text node.
        # What arrives here is the residue: a block at the start of the document
        # or directly after a heading, with no preceding section to join. In
        # paragraph mode the pre-pass never runs and this handles every block,
        # which is why _run_node_processing warns about that combination.
        if handling == "part_of_section":
            logger.log_action("  No section to join, passed through unchanged")
            return NodeResult(node=node, output=node.content, was_processed=False)

    saved = state.checkpoint.nodes.get(key) if state.checkpoint else None
    if saved is not None:
        logger.log_action("  Restored from the progress file")
        return NodeResult(
            node=node,
            output=saved["output"],
            similarity=saved["similarity"],
            was_processed=True,
            input_chars=saved["input_chars"],
            output_chars=saved["output_chars"],
            elapsed=saved["elapsed"],
        )

    # --- AI call ---
    prompt = f"{preset.node_prompt}\n\n{node.content}"
    messages: list[dict] | None = None
    if preset.include_preprocessed_context and preprocessed_context:
        messages = [
            {"role": "user", "content": preset.preprocessing_prompt},
            {"role": "assistant", "content": preprocessed_context},
        ]

    reply = _call_ai(prompt, config, preset, messages=messages, file_path=image_path)
    logger.log_ai(f"  AI responded ({_reply_stats(len(node.content), reply)})")

    similarity = _compute_similarity(node.content, reply.text, config, state, image_path)
    if similarity is not None:
        logger.log_metric(f"  Similarity: {similarity:.3f} {get_similarity_indicator(similarity)}")

    result = NodeResult(
        node=node,
        output=reply.text,
        similarity=similarity,
        was_processed=True,
        input_chars=len(node.content),
        output_chars=len(reply.text),
        elapsed=reply.elapsed,
    )
    if state.checkpoint:
        state.checkpoint.record_node(key, result)
    return result


def _merge_headings(nodes: list[Node], preset: Preset) -> list[Node]:
    """Pre-pass for merge_next: merge each heading into the text node that follows it.

    A heading with no text node right after it in the same file is left alone;
    _process_node passes it through.

    Args:
        nodes: The parsed node list.
        preset: Active preset (reads heading_handling).

    Returns:
        The node list with each merged pair replaced by one text node.
    """
    if preset.heading_handling != "merge_next":
        return nodes
    merged: list[Node] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if (
            node.node_type == "heading"
            and i + 1 < len(nodes)
            and nodes[i + 1].node_type == "text"
            and nodes[i + 1].source == node.source
        ):
            merged.append(Node(
                index=node.index,
                node_type="text",
                content=f"{node.content}\n\n{nodes[i + 1].content}",
                heading_prefix=node.content.strip(),
                source=node.source,
            ))
            i += 2
        else:
            merged.append(node)
            i += 1
    return merged


def _absorb_into_sections(nodes: list[Node], preset: Preset) -> list[Node]:
    """Pre-pass for part_of_section: absorb the blocks that follow a section into it.

    Only blocks whose own handling field asks for part_of_section are absorbed,
    and only in section mode. A block with no section before it in the same
    file is left alone; _process_node passes it through.

    Args:
        nodes: The node list after _merge_headings().
        preset: Active preset (reads node_mode and the handling fields).

    Returns:
        The node list with absorbed blocks folded into their section.
    """
    if preset.node_mode != "section" or "part_of_section" not in (
        preset.non_text_handling, preset.image_handling
    ):
        return nodes
    assembled: list[Node] = []
    j = 0
    while j < len(nodes):
        node = nodes[j]
        if node.node_type == "text":
            combined_content = node.content
            while (
                j + 1 < len(nodes)
                and nodes[j + 1].node_type not in ("text", "heading")
                and nodes[j + 1].source == node.source
                and _handling_for(nodes[j + 1], preset) == "part_of_section"
            ):
                j += 1
                combined_content += f"\n\n{nodes[j].content}"
            assembled.append(Node(
                index=node.index,
                node_type="text",
                content=combined_content,
                heading_prefix=node.heading_prefix,
                source=node.source,
            ))
        else:
            assembled.append(node)
        j += 1
    return assembled


def _log_phase_two_stats(node_results: list[NodeResult]) -> None:
    """Log the Phase 2 summary: sizes, ratio and AI time of the nodes sent to the AI."""
    processed = [r for r in node_results if r.was_processed]
    if not processed:
        logger.log_success("Node processing complete.")
        return
    total_in = sum(r.input_chars for r in processed)
    total_out = sum(r.output_chars for r in processed)
    ratio = total_out / total_in if total_in > 0 else 0.0
    total_time = sum(r.elapsed for r in processed)
    average_time = total_time / len(processed)
    logger.log_success(
        f"Node processing complete. "
        f"Total Phase 2 stats: in: {total_in}, out: {total_out} chars | ratio (out/in): {ratio:.3f} | "
        f"AI time: {_format_duration(total_time)} (avg {_format_duration(average_time)} per node)"
    )


def _run_node_processing(
    corpus: Corpus,
    preset: Preset,
    config: AppConfig,
    preprocessed: dict[str, str],
    checkpoint: Checkpoint | None = None,
) -> tuple[list[NodeResult], dict[str, str]]:
    """Run Phase 2: apply node_prompt to each node independently.

    Skipped if preset.node_prompt is empty.

    Parses every file of the corpus, runs the merge_next and part_of_section
    pre-passes, then locates the image files when image_handling attaches them,
    then sends each node.

    Args:
        corpus: The input documents; relative image paths resolve against its root.
        preset: Active preset.
        config: Application configuration.
        preprocessed: Phase 1 results by source file name (may be empty).
        checkpoint: Progress file to restore from and record into, or None.

    Returns:
        Tuple of (list of NodeResult, assembled results string by source file name).

    Raises:
        FileNotFoundError: If image_handling sends images and one is missing.
        urllib.error.URLError: If a remote image cannot be downloaded.
    """
    if not preset.node_prompt.strip():
        return [], {}

    logger.log_separator()
    logger.log_info(f"Phase 2: Node Processing ({preset.node_mode} mode)")

    if preset.node_mode == "paragraph":
        for field_name, label in (("non_text_handling", "non-text"), ("image_handling", "image")):
            if getattr(preset, field_name) == "part_of_section":
                logger.log_warning(
                    f"{field_name}='part_of_section' requires section mode; "
                    f"{label} nodes will pass through unchanged"
                )

    nodes = parse_corpus(corpus, preset.node_mode)
    nodes = _merge_headings(nodes, preset)
    nodes = _absorb_into_sections(nodes, preset)

    node_results: list[NodeResult] = []
    output_parts: dict[str, list[str]] = {source.name: [] for source in corpus.files}
    # Counted after both pre-passes: what they merged is one node, not two.
    total = len(nodes)
    logger.log_action(f"Found {total} nodes. Processing...")
    state = _RunState(similarity_enabled=preset.compute_similarity, checkpoint=checkpoint)
    current_source: str | None = None
    with tempfile.TemporaryDirectory(prefix="iter-images-") as download_dir:
        images: dict[int, Path] = {}
        if preset.image_handling == "standalone_nodes":
            images = resolve_images(nodes, corpus.root, Path(download_dir))
        image_digests = {index: file_digest(path) for index, path in images.items()}
        keys = node_keys(nodes, image_digests)
        for position, (node, key) in enumerate(zip(nodes, keys), 1):
            if corpus.is_folder and node.source != current_source:
                current_source = node.source
                logger.log_action(f"File: {node.source}")
            logger.log_action(f"Node {position}/{total} (§{node.index}) [{node.node_type}]")
            result = _process_node(
                node, preset, config, preprocessed.get(node.source, ""), state, key,
                images.get(node.index),
            )
            node_results.append(result)

            if result.output:
                parts = output_parts[node.source]
                sep = _build_separator(node, result.similarity, preset.show_similarity_separator)
                if sep:
                    parts.append(sep)
                if preset.interlace_input_output and result.was_processed:
                    interlaced = f"**Original:**\n{node.content}\n\n**Processed:**\n{result.output}"
                    parts.append(interlaced)
                else:
                    parts.append(result.output)

    _log_phase_two_stats(node_results)
    return node_results, {name: "\n\n".join(parts) for name, parts in output_parts.items()}


# ── Phase 3: Postprocessing ───────────────────────────────────────────────────

def _postprocess(
    document_text: str,
    preprocessed: str,
    results: str,
    preset: Preset,
    config: AppConfig,
    checkpoint: Checkpoint | None = None,
    source: str | None = "",
) -> str:
    """Make one Phase 3 call, or restore it from the progress file.

    Builds the prompt by concatenating the postprocessing prompt text with
    any selected attachments (original input, preprocessed, results).

    Args:
        document_text: Original input text the call covers.
        preprocessed: Phase 1 result for that input.
        results: Phase 2 assembled output for that input.
        preset: Active preset.
        config: Application configuration.
        checkpoint: Progress file to restore from and record into, or None.
        source: Source file name the call covers, or None for the whole corpus;
            the key of the call in the progress file.

    Returns:
        AI response string.
    """
    if checkpoint and source in checkpoint.postprocessed:
        logger.log_action("Postprocessing restored from the progress file.")
        return checkpoint.postprocessed[source]

    attachments = [
        (title, text)
        for attached, title, text in (
            (preset.postprocess_attach_input, "ORIGINAL DOCUMENT", document_text),
            (preset.postprocess_attach_preprocessed, "PREPROCESSED", preprocessed),
            (preset.postprocess_attach_results, "NODE RESULTS", results),
        )
        if attached and text
    ]
    parts = [preset.postprocessing_prompt]
    parts += [f"\n\n=== {title} ===\n{text}" for title, text in attachments]
    prompt = "\n".join(parts)

    reply = _call_ai(prompt, config, preset)
    attached_chars = sum(len(text) for _, text in attachments)
    logger.log_success(f"Postprocessing complete ({_reply_stats(attached_chars, reply)}).")
    if checkpoint:
        checkpoint.record_postprocessed(source, reply.text, reply.elapsed)
    return reply.text


def _run_postprocessing(
    corpus: Corpus,
    preprocessed: dict[str, str],
    results: dict[str, str],
    preset: Preset,
    config: AppConfig,
    checkpoint: Checkpoint | None = None,
) -> tuple[dict[str, str], str]:
    """Run Phase 3: consolidate the results, per file or for the whole corpus.

    Skipped if preset.postprocessing_prompt is empty. A folder with
    postprocessing_scope 'corpus' gets one call whose attachments join every
    file under its header; any other input gets one call per file.

    Args:
        corpus: The input documents.
        preprocessed: Phase 1 results by source file name.
        results: Phase 2 assembled output by source file name.
        preset: Active preset.
        config: Application configuration.
        checkpoint: Progress file to restore from and record into, or None.

    Returns:
        Tuple of (AI response by source file name, corpus-wide AI response);
        only one of the two is filled, and both are empty if skipped.
    """
    if not preset.postprocessing_prompt.strip():
        return {}, ""

    logger.log_separator()
    logger.log_info("Phase 3: Postprocessing")
    if corpus.is_folder and preset.postprocessing_scope == "corpus":
        if preset.postprocess_attach_input:
            logger.log_warning(
                "postprocessing_scope='corpus' with postprocess_attach_input attaches every "
                "input file to a single call, which may not fit the model's context"
            )

        def joined(texts: dict[str, str]) -> str:
            return join_by_file([(s.name, texts.get(s.name, "")) for s in corpus.files], headers=True)

        document_text = join_by_file([(s.name, s.text) for s in corpus.files], headers=True)
        corpus_post = _postprocess(
            document_text, joined(preprocessed), joined(results), preset, config, checkpoint, None
        )
        return {}, corpus_post

    postprocessed: dict[str, str] = {}
    for source in corpus.files:
        if corpus.is_folder:
            logger.log_action(f"File: {source.name}")
        postprocessed[source.name] = _postprocess(
            source.text, preprocessed.get(source.name, ""), results.get(source.name, ""),
            preset, config, checkpoint, source.name,
        )
    return postprocessed, ""


# ── Public API ────────────────────────────────────────────────────────────────

def run_iter(
    input_path: str | None = None,
    input_text: str | None = None,
    output_path: str | None = None,
    preset: Preset | None = None,
    config: AppConfig | None = None,
    context_file: str | None = None,
    on_mismatch: str = "ask",
) -> ProcessingResult:
    """Run the full Iter processing pipeline on a document or a folder of documents.

    Accepts a file path, a folder path, or a raw text string as input. A folder
    contributes the Markdown files (.md, .markdown) directly inside it, in
    natural order of their names. Runs up to three sequential phases
    (preprocessing, node processing, postprocessing), each of which is skipped
    when its corresponding prompt is empty.

    Optionally saves the main output, preprocessed, and postprocessed results
    to disk according to the preset's output settings.

    Every completed AI call is saved to a progress file, so a run that stops
    early resumes where it left off when started again. The file is deleted
    once all outputs are written; see checkpoint.progress_path() for where it lives.

    Args:
        input_path: Path to the input Markdown file, or to a folder of them.
            Used if input_text is None.
        input_text: Raw Markdown text to process (bypasses file reading).
        output_path: Where the main output is saved. For a folder input in the
            'mirror' output_layout, the folder that receives one file per input
            file. Optional.
        preset: Active Preset configuration. Defaults to a bare Preset().
        config: Application configuration. Defaults to AppConfig.load().
        context_file: Optional text file appended to the system prompt of every
            AI call, under '=== REFERENCE CONTEXT ==='.
        on_mismatch: What to do with a progress file written for a different
            input, context, preset or model: 'ask' raises CheckpointMismatchError
            so the caller can ask the user, 'continue' keeps the saved results,
            'restart' discards them.

    Returns:
        A ProcessingResult with all intermediate and final results.

    Raises:
        ValueError: If neither input_text nor an existing input_path is
            provided, the folder holds no Markdown file, the output of a folder
            is misplaced (see corpus.check_output_location()), or on_mismatch
            is not a known policy.
        FileNotFoundError: If context_file does not exist, or image_handling is
            'standalone_nodes' and a local image does not exist. Relative image
            paths are resolved against the input file's folder, the input
            folder, or the working directory for input_text.
        CheckpointMismatchError: If on_mismatch is 'ask' and the progress file
            belongs to a different run.
        urllib.error.URLError: If such an image is a URL that cannot be downloaded.
    """
    start = time.perf_counter()
    if config is None:
        config = AppConfig.load()
    if preset is None:
        preset = Preset()

    # --- Configure providers ---
    # Idempotent: a no-op when the interface already applied the same settings.
    config.apply_to_client()

    # --- Load input ---
    corpus = load_corpus(input_path, input_text)
    check_output_location(corpus, output_path, preset.output_layout)
    context = Path(context_file).read_text(encoding="utf-8") if context_file else ""

    logger.log_separator()
    logger.log_info(f"Iter - provider: {config.provider} | model: {config.model_name}")
    logger.log_info(f"Node mode: {preset.node_mode}")
    if context_file:
        logger.log_info(f"Reference context: {context_file}")

    checkpoint = Checkpoint(
        progress_path(output_path, input_path, input_text),
        fingerprint(corpus, preset, config, context),
        on_mismatch,
    )
    if checkpoint.resumed:
        logger.log_info(f"Resuming from {checkpoint.path}: {checkpoint.summary()}")

    try:
        proc_result = _run_phases(
            corpus, output_path, _with_reference_context(preset, context), config, checkpoint
        )
    except BaseException:
        logger.log_warning(f"Progress saved to {checkpoint.path}; run again to resume.")
        raise
    checkpoint.delete()

    logger.log_separator()
    logger.log_success(f"Iter processing complete in {_format_duration(time.perf_counter() - start)}.")
    return proc_result


def _run_phases(
    corpus: Corpus,
    output_path: str | None,
    preset: Preset,
    config: AppConfig,
    checkpoint: Checkpoint,
) -> ProcessingResult:
    """Run the three phases and write the outputs; the body of run_iter().

    Args:
        corpus: The loaded input.
        output_path: Path where the main output will be saved (a folder for a
            mirrored folder input), or None.
        preset: Active preset, with the reference context in its system prompt.
        config: Application configuration.
        checkpoint: The run's progress file.

    Returns:
        A ProcessingResult with all intermediate and final results.
    """
    # --- Phase 1: Preprocessing ---
    preprocessed = _run_preprocessing(corpus, preset, config, checkpoint)

    # --- Phase 2: Node Processing ---
    keys_saved_before = set(checkpoint.nodes)
    node_results, results = _run_node_processing(corpus, preset, config, preprocessed, checkpoint)
    redone = {checkpoint.nodes[key]["source"] for key in checkpoint.nodes.keys() - keys_saved_before}
    if redone:
        # Phase 3 consolidates the node results: once a node of a file was redone,
        # a saved Phase 3 result of that file, or of the whole corpus, no longer
        # matches them.
        for source in (*redone, None):
            checkpoint.postprocessed.pop(source, None)

    # --- Phase 3: Postprocessing ---
    postprocessed, corpus_post = _run_postprocessing(
        corpus, preprocessed, results, preset, config, checkpoint
    )

    # --- Build result object ---
    files = [
        FileResult(
            source=source.name,
            preprocessed=preprocessed.get(source.name, ""),
            node_results=[r for r in node_results if r.node.source == source.name],
            results=results.get(source.name, ""),
            postprocessed=postprocessed.get(source.name, ""),
        )
        for source in corpus.files
    ]
    headers = corpus.is_folder
    proc_result = ProcessingResult(
        preprocessed=join_by_file([(f.source, f.preprocessed) for f in files], headers),
        node_results=node_results,
        results=join_by_file([(f.source, f.results) for f in files], headers),
        postprocessed=corpus_post or join_by_file([(f.source, f.postprocessed) for f in files], headers),
        files=files,
    )

    # --- Save outputs ---
    if output_path:
        write_outputs(proc_result, corpus_post, Path(output_path), preset, corpus.is_folder)

    return proc_result
