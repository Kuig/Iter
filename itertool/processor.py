"""Core processing pipeline for Iter.

Implements the three-phase pipeline:
  Phase 1 - Preprocessing: single AI call on the full document.
  Phase 2 - Node Processing: same prompt applied independently to each node.
  Phase 3 - Postprocessing: single AI call on the assembled results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from unified_ai_client import call_ai
from unified_ai_client import get_embedding as _get_embedding_vec

from itertool.checkpoint import Checkpoint, fingerprint, node_keys, progress_path
from itertool.config import AppConfig
from itertool.models import (
    Node,
    NodeResult,
    Preset,
    ProcessingResult,
    cosine_similarity,
    get_similarity_indicator,
)
from itertool.markdown_utils import (
    parse_into_paragraphs,
    parse_into_sections,
    clean_ai_response,
)
import itertool.logger as logger


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
) -> _AiReply:
    """Make a single AI call and return the cleaned response with its timing.

    Args:
        prompt: User prompt text.
        config: Application configuration (provider, model, temperature, thinking).
        preset: Active preset (provides system_prompt).
        messages: Optional prior conversation history for context injection.

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
        temperature=config.temperature,
        thinking=config.thinking,
    )
    return _AiReply(
        text=clean_ai_response(response.text),
        elapsed=time.perf_counter() - start,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def _get_embedding(text: str, config: AppConfig) -> list[float]:
    """Compute a text embedding vector using the configured embedding model.

    Args:
        text: Text to embed.
        config: Application configuration (reads embedding_provider and embedding_model).

    Returns:
        Embedding vector as a list of floats.

    Raises:
        Exception: Whatever UnifiedAiClient raises once its retries are spent.
    """
    return _get_embedding_vec(
        provider=config.embedding_provider,
        model=config.embedding_model,
        text=text,
    )


def _compute_similarity(original: str, output: str, config: AppConfig, state: _RunState) -> float | None:
    """Return the cosine similarity of two texts, or None if it is not computed.

    The first embedding failure is logged once and disables similarity for the
    rest of the run.

    Args:
        original: Node content sent to the AI.
        output: AI response for that node.
        config: Application configuration (embedding provider and model).
        state: Run state; similarity_enabled is read and may be switched off.

    Returns:
        The similarity score, or None when disabled or failed.
    """
    if not state.similarity_enabled:
        return None
    try:
        vec1 = _get_embedding(original, config)
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

def _run_preprocessing(
    document_text: str,
    preset: Preset,
    config: AppConfig,
    checkpoint: Checkpoint | None = None,
) -> str:
    """Run Phase 1: a single AI call over the full document.

    Skipped if preset.preprocessing_prompt is empty.

    Args:
        document_text: Full input document text.
        preset: Active preset.
        config: Application configuration.
        checkpoint: Progress file to restore from and record into, or None.

    Returns:
        AI response string, or empty string if skipped.
    """
    if not preset.preprocessing_prompt.strip():
        return ""

    logger.log_separator()
    logger.log_info("Phase 1: Preprocessing")
    if checkpoint and checkpoint.preprocessed is not None:
        logger.log_action("Preprocessing restored from the progress file.")
        return checkpoint.preprocessed

    prompt = f"{preset.preprocessing_prompt}\n\n{document_text}"
    reply = _call_ai(prompt, config, preset)
    logger.log_success(f"Preprocessing complete ({_reply_stats(len(document_text), reply)}).")
    if checkpoint:
        checkpoint.record_preprocessed(reply.text, reply.elapsed)
    return reply.text


# ── Phase 2: Node Processing ──────────────────────────────────────────────────

def _resolve_nodes(document_text: str, preset: Preset) -> list[Node]:
    """Parse the document into nodes according to the preset's node_mode.

    Args:
        document_text: Full input document text.
        preset: Active preset (reads node_mode).

    Returns:
        List of Node objects.
    """
    if preset.node_mode == "section":
        return parse_into_sections(document_text)
    return parse_into_paragraphs(document_text)


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


def _process_node(
    node: Node,
    preset: Preset,
    config: AppConfig,
    preprocessed_context: str,
    state: _RunState,
    key: str = "",
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

    Returns:
        A NodeResult with the output and optional similarity score.
    """
    is_structural = node.node_type != "text"

    # --- Determine handling for structural (non-text) nodes ---
    if is_structural:
        handling = (
            preset.heading_handling
            if node.node_type == "heading"
            else preset.non_text_handling
        )

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

    reply = _call_ai(prompt, config, preset, messages=messages)
    logger.log_ai(f"  AI responded ({_reply_stats(len(node.content), reply)})")

    similarity = _compute_similarity(node.content, reply.text, config, state)
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


def _run_node_processing(
    document_text: str,
    preset: Preset,
    config: AppConfig,
    preprocessed: str,
    checkpoint: Checkpoint | None = None,
) -> tuple[list[NodeResult], str]:
    """Run Phase 2: apply node_prompt to each node independently.

    Skipped if preset.node_prompt is empty.

    Applies merge_next heading logic by looking ahead in the node list.
    Applies part_of_section non-text handling in section mode by re-assembling
    non-text content into the surrounding section text before AI call.

    Args:
        document_text: Full input document text.
        preset: Active preset.
        config: Application configuration.
        preprocessed: Phase 1 result (may be empty).
        checkpoint: Progress file to restore from and record into, or None.

    Returns:
        Tuple of (list of NodeResult, assembled results string).
    """
    if not preset.node_prompt.strip():
        return [], ""

    logger.log_separator()
    logger.log_info(f"Phase 2: Node Processing ({preset.node_mode} mode)")

    if preset.non_text_handling == "part_of_section" and preset.node_mode != "section":
        logger.log_warning(
            "non_text_handling='part_of_section' requires section mode; "
            "non-text nodes will pass through unchanged"
        )

    nodes = _resolve_nodes(document_text, preset)

    node_results: list[NodeResult] = []
    output_parts: list[str] = []

    # Pre-pass: merge each heading into the text node that follows it.
    merged_nodes: list[Node] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if (
            node.node_type == "heading"
            and preset.heading_handling == "merge_next"
            and i + 1 < len(nodes)
            and nodes[i + 1].node_type == "text"
        ):
            next_node = nodes[i + 1]
            merged_content = f"{node.content}\n\n{next_node.content}"
            merged = Node(
                index=node.index,
                node_type="text",
                content=merged_content,
                heading_prefix=node.content.strip(),
            )
            merged_nodes.append(merged)
            i += 2
        else:
            merged_nodes.append(node)
            i += 1

    # Pre-pass: absorb the non-text blocks that follow a section into it.
    if preset.node_mode == "section" and preset.non_text_handling == "part_of_section":
        assembled: list[Node] = []
        j = 0
        while j < len(merged_nodes):
            node = merged_nodes[j]
            if node.node_type == "text":
                combined_content = node.content
                while (
                    j + 1 < len(merged_nodes)
                    and merged_nodes[j + 1].node_type not in ("text", "heading")
                ):
                    j += 1
                    combined_content += f"\n\n{merged_nodes[j].content}"
                assembled.append(Node(
                    index=node.index,
                    node_type="text",
                    content=combined_content,
                    heading_prefix=node.heading_prefix,
                ))
            else:
                assembled.append(node)
            j += 1
        merged_nodes = assembled

    # Counted after both pre-passes: what they merged is one node, not two.
    total = len(merged_nodes)
    logger.log_action(f"Found {total} nodes. Processing...")
    state = _RunState(similarity_enabled=preset.compute_similarity, checkpoint=checkpoint)
    for position, (node, key) in enumerate(zip(merged_nodes, node_keys(merged_nodes)), 1):
        logger.log_action(f"Node {position}/{total} (§{node.index}) [{node.node_type}]")
        result = _process_node(node, preset, config, preprocessed, state, key)
        node_results.append(result)

        if result.output:
            sep = _build_separator(node, result.similarity, preset.show_similarity_separator)
            if sep:
                output_parts.append(sep)
            if preset.interlace_input_output and result.was_processed:
                interlaced = f"**Original:**\n{node.content}\n\n**Processed:**\n{result.output}"
                output_parts.append(interlaced)
            else:
                output_parts.append(result.output)

    results_text = "\n\n".join(output_parts)

    processed_results = [r for r in node_results if r.was_processed]
    if processed_results:
        total_in = sum(r.input_chars for r in processed_results)
        total_out = sum(r.output_chars for r in processed_results)
        ratio = total_out / total_in if total_in > 0 else 0.0
        total_time = sum(r.elapsed for r in processed_results)
        average_time = total_time / len(processed_results)
        logger.log_success(
            f"Node processing complete. "
            f"Total Phase 2 stats: in: {total_in}, out: {total_out} chars | ratio (out/in): {ratio:.3f} | "
            f"AI time: {_format_duration(total_time)} (avg {_format_duration(average_time)} per node)"
        )
    else:
        logger.log_success("Node processing complete.")

    return node_results, results_text


# ── Phase 3: Postprocessing ───────────────────────────────────────────────────

def _run_postprocessing(
    document_text: str,
    preprocessed: str,
    results: str,
    preset: Preset,
    config: AppConfig,
    checkpoint: Checkpoint | None = None,
) -> str:
    """Run Phase 3: a single AI call to consolidate all results.

    Skipped if preset.postprocessing_prompt is empty.

    Builds the prompt by concatenating the postprocessing prompt text with
    any selected attachments (original input, preprocessed, results).

    Args:
        document_text: Original input document text.
        preprocessed: Phase 1 result.
        results: Phase 2 assembled output.
        preset: Active preset.
        config: Application configuration.
        checkpoint: Progress file to restore from and record into, or None.

    Returns:
        AI response string, or empty string if skipped.
    """
    if not preset.postprocessing_prompt.strip():
        return ""

    logger.log_separator()
    logger.log_info("Phase 3: Postprocessing")
    if checkpoint and checkpoint.postprocessed is not None:
        logger.log_action("Postprocessing restored from the progress file.")
        return checkpoint.postprocessed

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
        checkpoint.record_postprocessed(reply.text, reply.elapsed)
    return reply.text


# ── Output assembly ───────────────────────────────────────────────────────────

def _compose_output(
    preprocessed: str,
    results: str,
    postprocessed: str,
    preset: Preset,
) -> str:
    """Build the main output by concatenating all selected sections in order.

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


# ── Public API ────────────────────────────────────────────────────────────────

def run_iter(
    input_file: str | None = None,
    input_text: str | None = None,
    output_file: str | None = None,
    preset: Preset | None = None,
    config: AppConfig | None = None,
    on_mismatch: str = "ask",
) -> ProcessingResult:
    """Run the full Iter processing pipeline on a document.

    Accepts either a file path or a raw text string as input. Runs up to
    three sequential phases (preprocessing, node processing, postprocessing),
    each of which is skipped when its corresponding prompt is empty.

    Optionally saves the main output, preprocessed, and postprocessed results
    to disk according to the preset's output settings.

    Every completed AI call is saved to a progress file, so a run that stops
    early resumes where it left off when started again. The file is deleted
    once all outputs are written; see checkpoint.progress_path() for where it lives.

    Args:
        input_file: Path to the input Markdown file. Used if input_text is None.
        input_text: Raw Markdown text to process (bypasses file reading).
        output_file: Path where the main output will be saved. Optional.
        preset: Active Preset configuration. Defaults to a bare Preset().
        config: Application configuration. Defaults to AppConfig.load().
        on_mismatch: What to do with a progress file written for a different
            input, preset or model: 'ask' raises CheckpointMismatchError so the
            caller can ask the user, 'continue' keeps the saved results,
            'restart' discards them.

    Returns:
        A ProcessingResult with all intermediate and final results.

    Raises:
        ValueError: If neither input_text nor a valid input_file is provided,
            or on_mismatch is not a known policy.
        CheckpointMismatchError: If on_mismatch is 'ask' and the progress file
            belongs to a different run.
    """
    start = time.perf_counter()
    if config is None:
        config = AppConfig.load()
    if preset is None:
        preset = Preset()

    # --- Configure providers ---
    # Idempotent: a no-op when the interface already applied the same settings.
    config.apply_to_client()

    # --- Load document ---
    if input_text is not None:
        document_text = input_text
    elif input_file is not None and Path(input_file).exists():
        logger.log_action(f"Loading document from: {input_file}")
        document_text = Path(input_file).read_text(encoding="utf-8")
    else:
        raise ValueError(
            f"Provide a valid input_text or an existing input_file. Got: {input_file}"
        )

    logger.log_separator()
    logger.log_info(f"Iter - provider: {config.provider} | model: {config.model_name}")
    logger.log_info(f"Node mode: {preset.node_mode}")

    checkpoint = Checkpoint(
        progress_path(output_file, input_file, input_text),
        fingerprint(document_text, preset, config),
        on_mismatch,
    )
    if checkpoint.resumed:
        logger.log_info(f"Resuming from {checkpoint.path}: {checkpoint.summary()}")

    try:
        proc_result = _run_phases(document_text, output_file, preset, config, checkpoint)
    except BaseException:
        logger.log_warning(f"Progress saved to {checkpoint.path}; run again to resume.")
        raise
    checkpoint.delete()

    logger.log_separator()
    logger.log_success(f"Iter processing complete in {_format_duration(time.perf_counter() - start)}.")
    return proc_result


def _run_phases(
    document_text: str,
    output_file: str | None,
    preset: Preset,
    config: AppConfig,
    checkpoint: Checkpoint,
) -> ProcessingResult:
    """Run the three phases and write the outputs; the body of run_iter().

    Args:
        document_text: The loaded input document.
        output_file: Path where the main output will be saved, or None.
        preset: Active preset.
        config: Application configuration.
        checkpoint: The run's progress file.

    Returns:
        A ProcessingResult with all intermediate and final results.
    """
    # --- Phase 1: Preprocessing ---
    preprocessed = _run_preprocessing(document_text, preset, config, checkpoint)

    # --- Phase 2: Node Processing ---
    nodes_saved_before = len(checkpoint.nodes)
    node_results, results = _run_node_processing(
        document_text, preset, config, preprocessed, checkpoint
    )
    if len(checkpoint.nodes) != nodes_saved_before:
        # Phase 3 consolidates the node results: once any node was redone, a
        # saved Phase 3 result no longer matches them.
        checkpoint.postprocessed = None

    # --- Phase 3: Postprocessing ---
    postprocessed = _run_postprocessing(
        document_text, preprocessed, results, preset, config, checkpoint
    )

    # --- Build result object ---
    proc_result = ProcessingResult(
        preprocessed=preprocessed,
        node_results=node_results,
        results=results,
        postprocessed=postprocessed,
    )

    # --- Save outputs ---
    main_output = _compose_output(preprocessed, results, postprocessed, preset)

    if output_file and main_output:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(main_output, encoding="utf-8")
        logger.log_save(f"Main output saved to: {output_file}")

    if preset.save_preprocessed_separately and preprocessed and output_file:
        pre_path = Path(output_file).with_stem(Path(output_file).stem + "_preprocessed")
        pre_path.write_text(preprocessed, encoding="utf-8")
        logger.log_save(f"Preprocessed output saved to: {pre_path}")

    if preset.save_postprocessed_separately and postprocessed and output_file:
        post_path = Path(output_file).with_stem(Path(output_file).stem + "_postprocessed")
        post_path.write_text(postprocessed, encoding="utf-8")
        logger.log_save(f"Postprocessed output saved to: {post_path}")

    return proc_result
