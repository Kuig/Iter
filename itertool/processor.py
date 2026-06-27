"""Core processing pipeline for Iter.

Implements the three-phase pipeline:
  Phase 1 — Preprocessing: single AI call on the full document.
  Phase 2 — Node Processing: same prompt applied independently to each node.
  Phase 3 — Postprocessing: single AI call on the assembled results.
"""
from __future__ import annotations

from pathlib import Path

from unified_ai_client import call_ai, configure_provider
from unified_ai_client import get_embedding as _get_embedding_vec

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

def _call_ai(
    prompt: str,
    config: AppConfig,
    preset: Preset,
    *,
    messages: list[dict] | None = None,
) -> str:
    """Make a single AI call and return the cleaned response text.

    Args:
        prompt: User prompt text.
        config: Application configuration (provider, model, temperature, thinking).
        preset: Active preset (provides system_prompt).
        messages: Optional prior conversation history for context injection.

    Returns:
        Cleaned AI response text.
    """
    system_prompt = preset.system_prompt or None
    response = call_ai(
        provider=config.provider,
        model=config.model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
        temperature=config.temperature,
        thinking=config.thinking,
    )
    return clean_ai_response(response.text)


def _get_embedding(text: str, config: AppConfig) -> list[float]:
    """Compute a text embedding vector using the configured embedding model.

    Args:
        text: Text to embed.
        config: Application configuration (reads embedding_provider and embedding_model).

    Returns:
        Embedding vector as a list of floats, or an empty list on failure.
    """
    try:
        return _get_embedding_vec(
            provider=config.embedding_provider,
            model=config.embedding_model,
            text=text,
        )
    except Exception as exc:
        logger.log_warning(f"Embedding request failed ({config.embedding_model}): {exc}")
        return []


# ── Phase 1: Preprocessing ────────────────────────────────────────────────────

def _run_preprocessing(
    document_text: str,
    preset: Preset,
    config: AppConfig,
) -> str:
    """Run Phase 1: a single AI call over the full document.

    Skipped if preset.preprocessing_prompt is empty.

    Args:
        document_text: Full input document text.
        preset: Active preset.
        config: Application configuration.

    Returns:
        AI response string, or empty string if skipped.
    """
    if not preset.preprocessing_prompt.strip():
        return ""

    logger.log_separator()
    logger.log_info("Phase 1: Preprocessing")
    prompt = f"{preset.preprocessing_prompt}\n\n{document_text}"
    input_len = len(document_text)
    result = _call_ai(prompt, config, preset)
    logger.log_success(f"Preprocessing complete (in: {input_len}, out: {len(result)} chars).")
    return result


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
    return " — ".join(parts)


def _process_node(
    node: Node,
    preset: Preset,
    config: AppConfig,
    preprocessed_context: str,
) -> NodeResult:
    """Process a single node through the AI with optional similarity computation.

    Applies heading/non-text handling rules before deciding whether to call AI.

    Args:
        node: The node to process.
        preset: Active preset (reads handling rules and options).
        config: Application configuration.
        preprocessed_context: Phase 1 result, used if include_preprocessed_context is True.

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
            logger.log_action(f"  Node {node.index} [{node.node_type}] — discarded")
            return NodeResult(node=node, output="", was_processed=False)

        if handling == "report_unprocessed":
            logger.log_action(f"  Node {node.index} [{node.node_type}] — passed through unchanged")
            return NodeResult(node=node, output=node.content, was_processed=False)

        if handling == "standalone_nodes":
            logger.log_action(f"  Node {node.index} [{node.node_type}] — processing as standalone")
            # Falls through to AI call below

        if handling == "merge_next":
            # merge_next is handled at the caller level (pre-merge), so this
            # case should not normally reach here. Pass through safely.
            logger.log_action(f"  Node {node.index} [{node.node_type}] — merge_next (passed through)")
            return NodeResult(node=node, output=node.content, was_processed=False)

        # part_of_section is handled at the caller level (section assembly)
        if handling == "part_of_section":
            logger.log_action(f"  Node {node.index} [{node.node_type}] — included in section (passed through)")
            return NodeResult(node=node, output=node.content, was_processed=False)

    # --- AI call ---
    prompt = f"{preset.node_prompt}\n\n{node.content}"
    messages: list[dict] | None = None
    if preset.include_preprocessed_context and preprocessed_context:
        messages = [
            {"role": "user", "content": preset.preprocessing_prompt},
            {"role": "assistant", "content": preprocessed_context},
        ]

    output = _call_ai(prompt, config, preset, messages=messages)
    
    # Calculate input length (only the node content characters, not the prompt payload)
    input_len = len(node.content)
    
    logger.log_ai(f"  Node {node.index} — AI responded (in: {input_len}, out: {len(output)} chars)")

    # --- Optional cosine similarity ---
    similarity: float | None = None
    if preset.compute_similarity:
        vec1 = _get_embedding(node.content, config)
        vec2 = _get_embedding(output, config)
        if vec1 and vec2:
            similarity = cosine_similarity(vec1, vec2)
            indicator = get_similarity_indicator(similarity)
            logger.log_metric(f"  Similarity: {similarity:.3f} {indicator}")

    return NodeResult(
        node=node,
        output=output,
        similarity=similarity,
        was_processed=True,
        input_chars=input_len,
        output_chars=len(output),
    )


def _run_node_processing(
    document_text: str,
    preset: Preset,
    config: AppConfig,
    preprocessed: str,
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

    Returns:
        Tuple of (list of NodeResult, assembled results string).
    """
    if not preset.node_prompt.strip():
        return [], ""

    logger.log_separator()
    logger.log_info(f"Phase 2: Node Processing ({preset.node_mode} mode)")

    nodes = _resolve_nodes(document_text, preset)
    logger.log_action(f"Found {len(nodes)} nodes. Processing...")

    node_results: list[NodeResult] = []
    output_parts: list[str] = []

    # Pre-pass: handle merge_next for headings (paragraph mode)
    # Build a merged version of the nodes list
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
            i += 2  # skip the next text node, it was merged
        else:
            merged_nodes.append(node)
            i += 1

    # Section mode: handle part_of_section for non-text blocks by re-assembling
    # If non_text_handling == part_of_section in section mode, non-text blocks
    # that appear between text sections are already part of the section node
    # (parse_into_sections already groups text paragraphs; structural nodes come
    # through individually). We handle this by detecting section-type nodes
    # adjacent to text nodes and merging them in.
    if preset.node_mode == "section" and preset.non_text_handling == "part_of_section":
        assembled: list[Node] = []
        j = 0
        while j < len(merged_nodes):
            node = merged_nodes[j]
            if node.node_type == "text":
                # Absorb any immediately following non-heading structural blocks
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

    for node in merged_nodes:
        logger.log_action(f"Node {node.index}/{len(merged_nodes)} [{node.node_type}]")
        result = _process_node(node, preset, config, preprocessed)
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
    
    # Calculate sums and ratio for processed nodes
    processed_results = [r for r in node_results if r.was_processed]
    if processed_results:
        total_in = sum(r.input_chars for r in processed_results)
        total_out = sum(r.output_chars for r in processed_results)
        ratio = total_out / total_in if total_in > 0 else 0.0
        logger.log_success(
            f"Node processing complete. "
            f"Total Phase 2 stats: in: {total_in}, out: {total_out} chars | ratio (out/in): {ratio:.3f}"
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

    Returns:
        AI response string, or empty string if skipped.
    """
    if not preset.postprocessing_prompt.strip():
        return ""

    logger.log_separator()
    logger.log_info("Phase 3: Postprocessing")

    parts = [preset.postprocessing_prompt]
    if preset.postprocess_attach_input and document_text:
        parts.append(f"\n\n=== ORIGINAL DOCUMENT ===\n{document_text}")
    if preset.postprocess_attach_preprocessed and preprocessed:
        parts.append(f"\n\n=== PREPROCESSED ===\n{preprocessed}")
    if preset.postprocess_attach_results and results:
        parts.append(f"\n\n=== NODE RESULTS ===\n{results}")

    prompt = "\n".join(parts)
    # Calculate input length (sum of the attached source text contents, excluding the prompt itself)
    attached_parts = []
    if preset.postprocess_attach_input and document_text:
        attached_parts.append(document_text)
    if preset.postprocess_attach_preprocessed and preprocessed:
        attached_parts.append(preprocessed)
    if preset.postprocess_attach_results and results:
        attached_parts.append(results)
    input_len = sum(len(p) for p in attached_parts)
    
    result = _call_ai(prompt, config, preset)
    logger.log_success(f"Postprocessing complete (in: {input_len}, out: {len(result)} chars).")
    return result


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
) -> ProcessingResult:
    """Run the full Iter processing pipeline on a document.

    Accepts either a file path or a raw text string as input. Runs up to
    three sequential phases (preprocessing, node processing, postprocessing),
    each of which is skipped when its corresponding prompt is empty.

    Optionally saves the main output, preprocessed, and postprocessed results
    to disk according to the preset's output settings.

    Args:
        input_file: Path to the input Markdown file. Used if input_text is None.
        input_text: Raw Markdown text to process (bypasses file reading).
        output_file: Path where the main output will be saved. Optional.
        preset: Active Preset configuration. Defaults to a bare Preset().
        config: Application configuration. Defaults to AppConfig.load().

    Returns:
        A ProcessingResult with all intermediate and final results.

    Raises:
        ValueError: If neither input_text nor a valid input_file is provided.
    """
    if config is None:
        config = AppConfig.load()
    if preset is None:
        preset = Preset()

    # --- Configure providers ---
    # Always register the Ollama base URL from config so UnifiedAiClient
    # uses the right server even when provider is not ollama (harmless no-op).
    configure_provider("ollama", url=config.ollama_url)

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
    logger.log_info(f"Iter — provider: {config.provider} | model: {config.model_name}")
    logger.log_info(f"Node mode: {preset.node_mode}")

    # --- Phase 1: Preprocessing ---
    preprocessed = _run_preprocessing(document_text, preset, config)

    # --- Phase 2: Node Processing ---
    node_results, results = _run_node_processing(
        document_text, preset, config, preprocessed
    )

    # --- Phase 3: Postprocessing ---
    postprocessed = _run_postprocessing(
        document_text, preprocessed, results, preset, config
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

    logger.log_separator()
    logger.log_success("Iter processing complete.")
    return proc_result
